from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ortools.sat.python import cp_model

app = FastAPI(title="School Timetable Solver - OR-Tools CP-SAT")

# --- Modèles de données ---
class Subject(BaseModel):
    id: str
    name: str
    hours_per_week: int
    min_block: int = 1
    max_block: int = 2
    single_session_only: bool = False

class Teacher(BaseModel):
    id: str
    name: str
    max_hours_per_day: int = 6

class ClassGroup(BaseModel):
    id: str
    name: str

class Room(BaseModel):
    id: str
    name: str

class TimetableRequest(BaseModel):
    classes: List[ClassGroup]
    subjects: List[Subject]
    teachers: List[Teacher]
    rooms: List[Room]
    days: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    slots_per_day: int = 8
    start_hour: int = 8
    max_time_seconds: int = 30

class TimetableEntry(BaseModel):
    class_id: str
    subject_id: str
    teacher_id: str
    room_id: str
    day: str
    start_hour: int
    duration: int

class TimetableResponse(BaseModel):
    status: str
    timetable: List[TimetableEntry]
    message: str

@app.post("/generate", response_model=TimetableResponse)
def generate_timetable(req: TimetableRequest):
    model = cp_model.CpModel()

    # --- 1. Préparation des créneaux ---
    all_slots = []
    for day in req.days:
        for h in range(req.start_hour, req.start_hour + req.slots_per_day):
            all_slots.append({"day": day, "hour": h})

    num_slots = len(all_slots)

    # Liste des besoins (Cours à placer)
    lesson_needs = []
    for cls in req.classes:
        for sub in req.subjects:
            if sub.hours_per_week > 0:
                lesson_needs.append({
                    "class_id": cls.id,
                    "subject_id": sub.id,
                    "hours": sub.hours_per_week,
                    "single": sub.single_session_only,
                    "min_b": sub.min_block,
                    "max_b": sub.max_block
                })

    # --- 2. Variables ---
    starts = {}
    durations = {}
    intervals = []

    # Map pour retrouver les intervalles par classe
    class_intervals = {cls.id: [] for cls in req.classes}

    for ln_idx, ln in enumerate(lesson_needs):
        for s_idx in range(num_slots):
            key = (ln_idx, s_idx)
            
            # Est-ce que le cours commence à cet index ?
            starts[key] = model.NewBoolVar(f'start_ln{ln_idx}_s{s_idx}')
            
            # Durée du bloc
            durations[key] = model.NewIntVar(ln["min_b"], ln["max_b"], f'dur_ln{ln_idx}_s{s_idx}')
            
            # Intervalle de temps (Optionnel : n'existe que si starts[key] == 1)
            suffix = f'interval_ln{ln_idx}_s{s_idx}'
            interval = model.NewOptionalIntervalVar(
                s_idx, durations[key], s_idx + durations[key], starts[key], suffix
            )
            
            intervals.append((ln_idx, s_idx, interval))
            class_intervals[ln["class_id"]].append(interval)

    # --- 3. Contraintes de Volume Horaire ---
    for ln_idx, ln in enumerate(lesson_needs):
        # La somme des durées de tous les blocs démarrés doit égaler les heures hebdo
        model.Add(sum(durations[(ln_idx, s)] * starts[(ln_idx, s)] for s in range(num_slots)) == ln["hours"])

        if ln["single"]:
            model.AddExactlyOne(starts[(ln_idx, s)] for s in range(num_slots))
            for s in range(num_slots):
                model.Add(durations[(ln_idx, s)] == ln["hours"]).OnlyEnforceIf(starts[(ln_idx, s)])

    # --- 4. Empêcher les cours de déborder sur le jour suivant ---
    for ln_idx, ln in enumerate(lesson_needs):
        for s_idx in range(num_slots):
            slot_in_day = s_idx % req.slots_per_day
            # La position de fin dans la journée ne doit pas dépasser le max de créneaux par jour
            model.Add(slot_in_day + durations[(ln_idx, s_idx)] <= req.slots_per_day).OnlyEnforceIf(starts[(ln_idx, s_idx)])

    # --- 5. Contrainte de Non-Chevauchement (Par Classe) ---
    for class_id in class_intervals:
        model.AddNoOverlap(class_intervals[class_id])

    # --- 6. Résolution ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = req.max_time_seconds
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        timetable_results = []
        for ln_idx, s_idx, interval in intervals:
            if solver.Value(starts[(ln_idx, s_idx)]):
                ln = lesson_needs[ln_idx]
                slot_info = all_slots[s_idx]
                
                timetable_results.append(TimetableEntry(
                    class_id=ln["class_id"],
                    subject_id=ln["subject_id"],
                    teacher_id=req.teachers[0].id if req.teachers else "N/A", # Simplifié
                    room_id=req.rooms[0].id if req.rooms else "N/A",       # Simplifié
                    day=slot_info["day"],
                    start_hour=slot_info["hour"],
                    duration=solver.Value(durations[(ln_idx, s_idx)])
                ))
        
        return TimetableResponse(
            status="success",
            timetable=timetable_results,
            message="Emploi du temps généré avec succès."
        )
    else:
        raise HTTPException(status_code=400, detail="Impossible de trouver une solution viable.")

@app.get("/health")
def health():
    return {"status": "online"}