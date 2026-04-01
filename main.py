"""
School Timetable Solver — OR-Tools CP-SAT
API FastAPI déployée sur Render
Contraintes strictes :
  - Un prof ne peut pas enseigner dans deux salles au même moment
  - Une classe ne peut pas avoir deux cours simultanés
  - Une salle ne peut pas accueillir deux événements simultanément
  - Respect de la pause déjeuner et des heures de travail
  - Max heures consécutives par enseignant
  - Indisponibilités enseignants
  - Assignation prof/matière/classe validée
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="School Timetable Solver — OR-Tools CP-SAT",
    description="Générateur d'emplois du temps scolaires par programmation par contraintes",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

class Subject(BaseModel):
    id: str
    name: str
    hours_per_week: int = Field(ge=1, le=40)
    preferred_block_minutes: int = Field(default=60, ge=30, le=240)
    is_single_session_only: bool = False

class Teacher(BaseModel):
    id: str
    name: str
    max_hours_per_day: int = Field(default=6, ge=1, le=12)
    max_hours_per_week: int = Field(default=30, ge=1, le=60)
    # Liste de créneaux indisponibles : [{"day": 1, "hour": 8}, ...]
    unavailable_slots: List[Dict[str, Any]] = []

class Room(BaseModel):
    id: str
    name: str
    capacity: int = Field(default=30, ge=1)
    room_type: str = "classroom"  # classroom, lab, computer_lab, etc.

class ClassGroup(BaseModel):
    id: str
    name: str
    size: int = Field(default=30, ge=1)

class TeacherAssignment(BaseModel):
    """Liaison explicite enseignant → classe → matière"""
    teacher_id: str
    class_id: str
    subject_id: str

class GlobalConstraints(BaseModel):
    period_start: str = "07:00"   # HH:MM
    period_end: str = "18:00"     # HH:MM
    lunch_start: str = "12:00"    # HH:MM
    lunch_end: str = "13:00"      # HH:MM
    max_consecutive_hours: int = Field(default=4, ge=1, le=8)
    days_of_week: List[int] = [1, 2, 3, 4, 5]  # 1=Lun, 5=Ven, 6=Sam

class TimetableRequest(BaseModel):
    classes: List[ClassGroup]
    subjects: List[Subject]
    teachers: List[Teacher]
    rooms: List[Room]
    assignments: List[TeacherAssignment]  # qui enseigne quoi à qui
    constraints: Optional[GlobalConstraints] = None
    max_time_seconds: int = Field(default=60, ge=5, le=300)

class TimetableEntry(BaseModel):
    class_id: str
    class_name: str
    subject_id: str
    subject_name: str
    teacher_id: str
    teacher_name: str
    room_id: str
    room_name: str
    room_type: str
    day: int          # 1=Lundi…6=Samedi
    start_hour: int   # heure entière (ex : 8)
    start_minute: int # 0 ou 30
    duration_minutes: int

class ConflictInfo(BaseModel):
    type: str
    message: str

class TimetableResponse(BaseModel):
    status: str
    timetable: List[TimetableEntry]
    conflicts: List[str] = []
    stats: Dict[str, Any] = {}
    message: str

class ConflictCheckRequest(BaseModel):
    """Validation d'un déplacement drag-and-drop"""
    slot_id: Optional[str] = None  # créneau qu'on déplace (pour l'exclure)
    class_id: str
    teacher_id: str
    room_id: Optional[str] = None
    day_of_week: int
    start_time: str  # HH:MM
    end_time: str    # HH:MM
    existing_slots: List[Dict[str, Any]] = []  # tous les créneaux actuels

class ConflictCheckResponse(BaseModel):
    conflicts: List[ConflictInfo]
    valid: bool

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def parse_hhmm(t: str) -> int:
    """Convertit HH:MM en minutes depuis minuit"""
    h, m = map(int, t.split(":"))
    return h * 60 + m

def minutes_to_hh(m: int) -> tuple[int, int]:
    return m // 60, m % 60

def time_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Vrai si deux intervalles [s1,e1) et [s2,e2) se chevauchent"""
    return s1 < e2 and e1 > s2

# ---------------------------------------------------------------------------
# Endpoint principal : génération OR-Tools CP-SAT
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=TimetableResponse)
def generate_timetable(req: TimetableRequest):
    logger.info(f"Génération demandée : {len(req.classes)} classes, {len(req.teachers)} profs, {len(req.rooms)} salles")

    cst = req.constraints or GlobalConstraints()

    # ---- Paramètres temporels ----
    PERIOD_START = parse_hhmm(cst.period_start)   # en minutes
    PERIOD_END   = parse_hhmm(cst.period_end)
    LUNCH_START  = parse_hhmm(cst.lunch_start)
    LUNCH_END    = parse_hhmm(cst.lunch_end)
    SLOT_MINUTES = 30   # granularité : 30 min
    DAYS         = sorted(cst.days_of_week)

    # ---- Créneaux disponibles dans une journée ----
    day_slots: List[int] = []  # liste de minutes-dans-la-journée
    t = PERIOD_START
    while t + SLOT_MINUTES <= PERIOD_END:
        if not (t >= LUNCH_START and t < LUNCH_END):
            day_slots.append(t)
        t += SLOT_MINUTES
    NUM_DAY_SLOTS = len(day_slots)

    # slot global = (jour_idx, slot_idx_dans_le_jour)
    def global_slot(day_idx: int, ds_idx: int) -> int:
        return day_idx * NUM_DAY_SLOTS + ds_idx

    TOTAL_SLOTS = len(DAYS) * NUM_DAY_SLOTS

    # ---- Index des entités ----
    class_map   = {c.id: c for c in req.classes}
    subject_map = {s.id: s for s in req.subjects}
    teacher_map = {t.id: t for t in req.teachers}
    room_map    = {r.id: r for r in req.rooms}

    # ---- Construire les besoins d'enseignement ----
    # Un besoin = (class_id, subject_id, teacher_id, hours_needed, block_slots)
    needs = []
    for asgn in req.assignments:
        if asgn.class_id not in class_map:
            continue
        if asgn.subject_id not in subject_map:
            continue
        if asgn.teacher_id not in teacher_map:
            continue
        subj = subject_map[asgn.subject_id]
        block_slots = max(1, subj.preferred_block_minutes // SLOT_MINUTES)
        # On limite le bloc à ce qui tient dans la journée
        block_slots = min(block_slots, NUM_DAY_SLOTS)
        total_slots_needed = (subj.hours_per_week * 60) // SLOT_MINUTES
        needs.append({
            "class_id":    asgn.class_id,
            "subject_id":  asgn.subject_id,
            "teacher_id":  asgn.teacher_id,
            "total":       total_slots_needed,   # créneaux 30-min à placer
            "block":       block_slots,           # taille d'un bloc
            "single":      subj.is_single_session_only,
        })

    if not needs:
        raise HTTPException(status_code=400, detail="Aucun besoin d'enseignement valide (vérifiez les assignations)")

    # ---- Modèle CP-SAT ----
    model = cp_model.CpModel()

    # Variables booléennes : start_vars[need_idx][slot] = 1 si le cours COMMENCE ici
    start_vars: Dict[int, Dict[int, cp_model.IntVar]] = {}
    # Variables de durée de bloc (nombre de créneaux 30-min)
    dur_vars: Dict[int, Dict[int, cp_model.IntVar]] = {}
    # Variables de salle choisie (index dans req.rooms)
    room_vars: Dict[int, Dict[int, cp_model.IntVar]] = {}

    for ni, need in enumerate(needs):
        start_vars[ni] = {}
        dur_vars[ni] = {}
        room_vars[ni] = {}
        block = need["block"]
        num_rooms = len(req.rooms)

        for day_idx, day in enumerate(DAYS):
            for ds_idx in range(NUM_DAY_SLOTS):
                gs = global_slot(day_idx, ds_idx)

                # Vérifier que le bloc tient dans la journée
                # (ne déborde pas sur le lendemain et ne chevauche pas la pause)
                block_fits = True
                for bk in range(block):
                    next_ds = ds_idx + bk
                    if next_ds >= NUM_DAY_SLOTS:
                        block_fits = False
                        break
                    slot_time = day_slots[next_ds]
                    # Pause déjeuner déjà exclue de day_slots, mais on vérifie
                    # qu'il n'y a pas de "trou" créé par la pause au milieu du bloc
                    if bk > 0:
                        expected = day_slots[ds_idx] + bk * SLOT_MINUTES
                        if slot_time != expected:
                            block_fits = False
                            break
                if not block_fits:
                    continue

                sv = model.NewBoolVar(f"s_n{ni}_g{gs}")
                dv = model.NewIntVar(block, block, f"d_n{ni}_g{gs}")  # durée fixe = bloc
                rv = model.NewIntVar(0, num_rooms - 1, f"r_n{ni}_g{gs}")

                start_vars[ni][gs] = sv
                dur_vars[ni][gs] = dv
                room_vars[ni][gs] = rv

    # ---- Contrainte : volume horaire total par besoin ----
    for ni, need in enumerate(needs):
        valid_gs = list(start_vars[ni].keys())
        if not valid_gs:
            logger.warning(f"Besoin {ni} ({need['subject_id']}/{need['class_id']}) : aucun créneau disponible !")
            continue
        block = need["block"]
        # Somme des durées de blocs activés = total créneaux demandés
        model.Add(
            sum(dur_vars[ni][gs] * start_vars[ni][gs] for gs in valid_gs) == need["total"]
        )
        if need["single"]:
            model.AddExactlyOne(start_vars[ni][gs] for gs in valid_gs)

    # ---- Contrainte : la salle ne peut servir qu'un seul cours à la fois ----
    # Pour chaque salle r et chaque créneau 30-min gs :
    #   sum(start_vars[ni][gs'] si gs' dans le bloc couvrant gs et room_vars[ni][gs']==r) <= 1
    #
    # On utilise AddNoOverlap sur des intervalles optionnels par salle.
    # Mais comme la salle est une variable, on crée des var booléennes room_chosen[ni][gs][r]

    # room_chosen[ni][gs][r] = 1 ssi start_vars[ni][gs] == 1 et room_vars[ni][gs] == r
    room_chosen: Dict[int, Dict[int, Dict[int, cp_model.IntVar]]] = {}
    for ni, need in enumerate(needs):
        room_chosen[ni] = {}
        for gs, sv in start_vars[ni].items():
            room_chosen[ni][gs] = {}
            for ri in range(len(req.rooms)):
                rc = model.NewBoolVar(f"rc_n{ni}_g{gs}_r{ri}")
                room_chosen[ni][gs][ri] = rc
                # rc == 1 ssi sv == 1 et room_vars[ni][gs] == ri
                b_eq = model.NewBoolVar(f"b_eq_n{ni}_g{gs}_r{ri}")
                model.Add(room_vars[ni][gs] == ri).OnlyEnforceIf(b_eq)
                model.Add(room_vars[ni][gs] != ri).OnlyEnforceIf(b_eq.Not())
                model.AddBoolAnd([sv, b_eq]).OnlyEnforceIf(rc)
                model.AddBoolOr([sv.Not(), b_eq.Not()]).OnlyEnforceIf(rc.Not())

    # Pour chaque créneau 30-min dans le calendrier et chaque salle :
    # au maximum 1 cours peut utiliser cette salle à ce moment
    # Un cours ni démarrant à gs couvre les créneaux gs..gs+block-1
    # On indexe le chevauchement par (day_idx, ds_idx) × salle

    for day_idx in range(len(DAYS)):
        for ds_idx in range(NUM_DAY_SLOTS):
            gs_here = global_slot(day_idx, ds_idx)
            for ri in range(len(req.rooms)):
                occupiers = []
                for ni, need in enumerate(needs):
                    block = need["block"]
                    for bk in range(block):
                        gs_start = global_slot(day_idx, ds_idx - bk)
                        if gs_start in room_chosen[ni] and ri in room_chosen[ni][gs_start]:
                            # Vérifier que ce bloc couvre bien ds_idx
                            if ds_idx - bk >= 0:
                                expected_end_ds = (ds_idx - bk) + block
                                if ds_idx < expected_end_ds and ds_idx - bk >= 0:
                                    occupiers.append(room_chosen[ni][gs_start][ri])
                if len(occupiers) > 1:
                    model.Add(sum(occupiers) <= 1)

    # ---- Contrainte : un professeur ne peut pas enseigner dans 2 endroits simultanément ----
    for day_idx in range(len(DAYS)):
        for ds_idx in range(NUM_DAY_SLOTS):
            # Par prof : au max 1 cours actif à ce créneau
            teacher_active: Dict[str, List[cp_model.IntVar]] = {}
            for ni, need in enumerate(needs):
                tid = need["teacher_id"]
                block = need["block"]
                if tid not in teacher_active:
                    teacher_active[tid] = []
                for bk in range(block):
                    ds_start = ds_idx - bk
                    if ds_start < 0:
                        continue
                    gs_start = global_slot(day_idx, ds_start)
                    if gs_start in start_vars[ni]:
                        # Ce cours (ni) démarrant à gs_start couvre ds_idx
                        expected_end_ds = ds_start + block
                        if ds_idx < expected_end_ds:
                            teacher_active[tid].append(start_vars[ni][gs_start])
            for tid, acts in teacher_active.items():
                if len(acts) > 1:
                    model.Add(sum(acts) <= 1)

    # ---- Contrainte : une classe ne peut pas avoir 2 cours simultanément ----
    for day_idx in range(len(DAYS)):
        for ds_idx in range(NUM_DAY_SLOTS):
            class_active: Dict[str, List[cp_model.IntVar]] = {}
            for ni, need in enumerate(needs):
                cid = need["class_id"]
                block = need["block"]
                if cid not in class_active:
                    class_active[cid] = []
                for bk in range(block):
                    ds_start = ds_idx - bk
                    if ds_start < 0:
                        continue
                    gs_start = global_slot(day_idx, ds_start)
                    if gs_start in start_vars[ni]:
                        expected_end_ds = ds_start + block
                        if ds_idx < expected_end_ds:
                            class_active[cid].append(start_vars[ni][gs_start])
            for cid, acts in class_active.items():
                if len(acts) > 1:
                    model.Add(sum(acts) <= 1)

    # ---- Contrainte : indisponibilités des enseignants ----
    for ni, need in enumerate(needs):
        tid = need["teacher_id"]
        teacher = teacher_map.get(tid)
        if not teacher:
            continue
        for unavail in teacher.unavailable_slots:
            uday = unavail.get("day")   # 1-6
            uhour = unavail.get("hour") # heure entière ou None (= journée entière)
            if uday not in DAYS:
                continue
            day_idx = DAYS.index(uday)
            for ds_idx, slot_min in enumerate(day_slots):
                if uhour is None or slot_min // 60 == uhour:
                    gs = global_slot(day_idx, ds_idx)
                    if gs in start_vars[ni]:
                        model.Add(start_vars[ni][gs] == 0)

    # ---- Contrainte : max heures par jour par enseignant ----
    for ni, need in enumerate(needs):
        tid = need["teacher_id"]
        teacher = teacher_map.get(tid)
        if not teacher:
            continue
        max_per_day_slots = (teacher.max_hours_per_day * 60) // SLOT_MINUTES
        block = need["block"]
        for day_idx in range(len(DAYS)):
            day_starts = [
                start_vars[ni][global_slot(day_idx, ds_idx)]
                for ds_idx in range(NUM_DAY_SLOTS)
                if global_slot(day_idx, ds_idx) in start_vars[ni]
            ]
            if day_starts:
                # somme des blocs démarrés ce jour * taille bloc <= max
                model.Add(sum(day_starts) * block <= max_per_day_slots)

    # ---- Contrainte : heures consécutives max par enseignant ----
    max_consec_slots = (cst.max_consecutive_hours * 60) // SLOT_MINUTES
    for day_idx in range(len(DAYS)):
        # Pour chaque fenêtre de (max_consec_slots + 1) créneaux contigus
        for ds_start in range(NUM_DAY_SLOTS - max_consec_slots):
            # Fenêtre : ds_start..ds_start+max_consec_slots (inclus)
            # Si tous les créneaux de la fenêtre étendue sont occupés par le même prof → violation
            teacher_window: Dict[str, List[cp_model.IntVar]] = {}
            for ni, need in enumerate(needs):
                tid = need["teacher_id"]
                if tid not in teacher_window:
                    teacher_window[tid] = []
                block = need["block"]
                # Tous les créneaux ds dans la fenêtre élargie
                for ds_check in range(ds_start, ds_start + max_consec_slots + 1):
                    for bk in range(block):
                        ds_s = ds_check - bk
                        if ds_s < 0:
                            continue
                        gs = global_slot(day_idx, ds_s)
                        if gs in start_vars[ni]:
                            end_ds = ds_s + block
                            if ds_check < end_ds:
                                teacher_window[tid].append(start_vars[ni][gs])
            # Pour chaque prof, on ne peut pas avoir max_consec+1 créneaux actifs de suite
            # (approximation : limiter la somme dans la fenêtre)
            for tid, acts in teacher_window.items():
                if len(acts) > max_consec_slots:
                    model.Add(sum(acts) <= max_consec_slots)

    # ---- Objectif : minimiser les trous dans les emplois du temps ----
    # (optionnel, aide à compacter les horaires)
    # On essaie de regrouper les cours en début de journée
    objective_terms = []
    for ni, need in enumerate(needs):
        for day_idx in range(len(DAYS)):
            for ds_idx in range(NUM_DAY_SLOTS):
                gs = global_slot(day_idx, ds_idx)
                if gs in start_vars[ni]:
                    # Pénaliser les créneaux tardifs
                    objective_terms.append(ds_idx * start_vars[ni][gs])
    if objective_terms:
        model.Minimize(sum(objective_terms))

    # ---- Résolution ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = req.max_time_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    logger.info(f"Résolution CP-SAT (timeout={req.max_time_seconds}s)...")
    status = solver.Solve(model)
    logger.info(f"Statut : {solver.StatusName(status)}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        detail = "Impossible de trouver une solution viable. "
        if status == cp_model.INFEASIBLE:
            detail += "Le problème est infaisable — vérifiez les contraintes et le nombre de salles/créneaux disponibles."
        else:
            detail += "Timeout atteint sans solution — réduisez le nombre de contraintes ou augmentez max_time_seconds."
        raise HTTPException(status_code=400, detail=detail)

    # ---- Extraction des résultats ----
    results: List[TimetableEntry] = []
    placed_hours: Dict[str, int] = {}
    missing: List[str] = []

    for ni, need in enumerate(needs):
        block = need["block"]
        placed = 0
        for day_idx, day in enumerate(DAYS):
            for ds_idx in range(NUM_DAY_SLOTS):
                gs = global_slot(day_idx, ds_idx)
                if gs not in start_vars[ni]:
                    continue
                if solver.Value(start_vars[ni][gs]) == 1:
                    ri = solver.Value(room_vars[ni][gs])
                    room = req.rooms[ri] if ri < len(req.rooms) else None
                    slot_min = day_slots[ds_idx]
                    h, m = minutes_to_hh(slot_min)
                    cls     = class_map.get(need["class_id"])
                    subj    = subject_map.get(need["subject_id"])
                    teacher = teacher_map.get(need["teacher_id"])
                    results.append(TimetableEntry(
                        class_id=need["class_id"],
                        class_name=cls.name if cls else need["class_id"],
                        subject_id=need["subject_id"],
                        subject_name=subj.name if subj else need["subject_id"],
                        teacher_id=need["teacher_id"],
                        teacher_name=teacher.name if teacher else need["teacher_id"],
                        room_id=room.id if room else "none",
                        room_name=room.name if room else "—",
                        room_type=room.room_type if room else "classroom",
                        day=day,
                        start_hour=h,
                        start_minute=m,
                        duration_minutes=block * SLOT_MINUTES,
                    ))
                    placed += block

        # Vérifier si toutes les heures ont été placées
        key = f"{need['class_id']}/{need['subject_id']}"
        placed_hours[key] = placed
        needed = need["total"]
        if placed < needed:
            subj = subject_map.get(need["subject_id"])
            cls  = class_map.get(need["class_id"])
            missing.append(
                f"{subj.name if subj else need['subject_id']} "
                f"({cls.name if cls else need['class_id']}) : "
                f"{(needed - placed) * SLOT_MINUTES // 60}h non placées"
            )

    stats = {
        "solver_status": solver.StatusName(status),
        "wall_time_s": round(solver.WallTime(), 2),
        "num_entries": len(results),
        "missing_hours": missing,
        "num_classes": len(req.classes),
        "num_teachers": len(req.teachers),
        "num_rooms": len(req.rooms),
    }

    return TimetableResponse(
        status="success" if not missing else "partial",
        timetable=results,
        conflicts=missing,
        stats=stats,
        message=(
            "Emploi du temps généré avec succès." if not missing
            else f"Généré avec {len(missing)} contrainte(s) non satisfaite(s) — voir 'conflicts'."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint : validation contraintes pour drag & drop
# ---------------------------------------------------------------------------

@app.post("/validate-move", response_model=ConflictCheckResponse)
def validate_move(req: ConflictCheckRequest):
    """
    Vérifie si déplacer un créneau vers un nouveau jour/horaire
    crée un conflit avec les créneaux existants.
    Retourne la liste des conflits. Si vide → drop autorisé.
    """
    start = parse_hhmm(req.start_time)
    end   = parse_hhmm(req.end_time)
    conflicts: List[ConflictInfo] = []

    for slot in req.existing_slots:
        # Exclure le créneau lui-même qu'on est en train de déplacer
        if req.slot_id and slot.get("id") == req.slot_id:
            continue
        # Même jour ?
        if slot.get("day_of_week") != req.day_of_week:
            continue

        s_start = parse_hhmm(slot.get("start_time", "00:00"))
        s_end   = parse_hhmm(slot.get("end_time", "00:00"))

        if not time_overlap(start, end, s_start, s_end):
            continue

        # Conflit professeur
        if slot.get("teacher_id") == req.teacher_id:
            conflicts.append(ConflictInfo(
                type="teacher",
                message=f"L'enseignant a déjà un cours de {slot.get('start_time', '')} à {slot.get('end_time', '')}",
            ))

        # Conflit classe
        if slot.get("class_id") == req.class_id:
            conflicts.append(ConflictInfo(
                type="class",
                message=f"La classe a déjà un cours de {slot.get('start_time', '')} à {slot.get('end_time', '')}",
            ))

        # Conflit salle
        if req.room_id and slot.get("room_id") == req.room_id:
            conflicts.append(ConflictInfo(
                type="room",
                message=f"La salle est déjà occupée de {slot.get('start_time', '')} à {slot.get('end_time', '')}",
            ))

    return ConflictCheckResponse(
        conflicts=conflicts,
        valid=len(conflicts) == 0,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "online",
        "solver": "OR-Tools CP-SAT",
        "version": "2.0.0",
    }

@app.get("/")
def root():
    return {
        "name": "School Timetable Solver",
        "endpoints": ["/generate", "/validate-move", "/health", "/docs"],
    }
