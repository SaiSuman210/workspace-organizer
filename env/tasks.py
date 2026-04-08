"""Task definitions for the Workspace Organizer benchmark.

Three tasks of increasing difficulty are defined here and registered
in the TASKS dict keyed by lowercase name.
"""
from __future__ import annotations

from dataclasses import dataclass

from env.models import File, TaskSolution


@dataclass
class Task:
    name: str
    instruction: str
    initial_files: list[File]
    initial_folders: dict[str, list[str]]
    solution: TaskSolution


# ---------------------------------------------------------------------------
# Task registry — populated at module load time (see bottom of file)
# ---------------------------------------------------------------------------
TASKS: dict[str, Task] = {}


def _register_tasks() -> None:
    """Build and register all task instances."""

    # ------------------------------------------------------------------
    # EASY: 5 files, rename-only solution
    # ------------------------------------------------------------------
    easy_files = [
        File(id="e001", name="IMG_001.JPG", type="image", date="2024-01-10",
             summary="beach vacation photo", size=2048000),
        File(id="e002", name="IMG_002.JPG", type="image", date="2024-01-11",
             summary="mountain hiking photo", size=1920000),
        File(id="e003", name="DOC_SCAN_A.PDF", type="document", date="2024-02-05",
             summary="scanned invoice", size=512000),
        File(id="e004", name="NOTES_DRAFT.TXT", type="document", date="2024-03-01",
             summary="meeting notes draft", size=8192),
        File(id="e005", name="PHOTO_BACKUP.PNG", type="image", date="2024-03-15",
             summary="family portrait backup", size=3145728),
    ]
    easy_folders: dict[str, list[str]] = {
        "root": ["e001", "e002", "e003", "e004", "e005"],
    }
    easy_solution = TaskSolution(
        expected_renames={
            "e001": "photo_001.jpg",
            "e002": "photo_002.jpg",
            "e003": "invoice_scan.pdf",
            "e004": "meeting_notes.txt",
            "e005": "family_portrait.png",
        },
        expected_placements={},
        expected_deletions=set(),
    )
    TASKS["easy"] = Task(
        name="easy",
        instruction=(
            "Clean up the messy file names in your Downloads folder. "
            "Rename each file to a lowercase, descriptive name that reflects its content "
            "(e.g., 'IMG_001.JPG' → 'photo_001.jpg'). No folders need to be created."
        ),
        initial_files=easy_files,
        initial_folders=easy_folders,
        solution=easy_solution,
    )

    # ------------------------------------------------------------------
    # MEDIUM: 10 files, create_folder + move solution
    # ------------------------------------------------------------------
    medium_files = [
        File(id="m001", name="tax_return_2022.pdf", type="document", date="2023-04-01",
             summary="2022 tax return document", size=204800),
        File(id="m002", name="tax_return_2023.pdf", type="document", date="2024-04-01",
             summary="2023 tax return document", size=215040),
        File(id="m003", name="w2_form_2023.pdf", type="document", date="2024-01-31",
             summary="W-2 wage statement 2023", size=102400),
        File(id="m004", name="paris_trip_day1.jpg", type="image", date="2023-06-10",
             summary="Paris Eiffel Tower photo", size=4096000),
        File(id="m005", name="paris_trip_day2.jpg", type="image", date="2023-06-11",
             summary="Paris Louvre museum photo", size=3840000),
        File(id="m006", name="paris_trip_day3.jpg", type="image", date="2023-06-12",
             summary="Paris Seine river photo", size=3584000),
        File(id="m007", name="rome_colosseum.jpg", type="image", date="2023-08-20",
             summary="Rome Colosseum photo", size=4200000),
        File(id="m008", name="rome_forum.jpg", type="image", date="2023-08-21",
             summary="Rome Forum photo", size=3900000),
        File(id="m009", name="receipt_hotel_paris.pdf", type="document", date="2023-06-15",
             summary="Paris hotel receipt", size=51200),
        File(id="m010", name="receipt_flight_rome.pdf", type="document", date="2023-08-18",
             summary="Rome flight receipt", size=48640),
    ]
    medium_folders: dict[str, list[str]] = {
        "root": [f.id for f in medium_files],
    }
    medium_solution = TaskSolution(
        expected_renames={},
        expected_placements={
            "m001": "tax_documents",
            "m002": "tax_documents",
            "m003": "tax_documents",
            "m004": "travel_photos",
            "m005": "travel_photos",
            "m006": "travel_photos",
            "m007": "travel_photos",
            "m008": "travel_photos",
            "m009": "travel_receipts",
            "m010": "travel_receipts",
        },
        expected_deletions=set(),
    )
    TASKS["medium"] = Task(
        name="medium",
        instruction=(
            "Organize your Downloads folder by creating category folders and moving files into them. "
            "Create folders for 'tax_documents', 'travel_photos', and 'travel_receipts', "
            "then move each file into the appropriate folder based on its content."
        ),
        initial_files=medium_files,
        initial_folders=medium_folders,
        solution=medium_solution,
    )

    # ------------------------------------------------------------------
    # HARD: 12 files including ≥2 duplicate pairs, folder + delete solution
    # ------------------------------------------------------------------
    hard_files = [
        # Unique files
        File(id="h001", name="project_proposal.docx", type="document", date="2024-01-05",
             summary="Q1 project proposal", size=307200),
        File(id="h002", name="budget_2024.xlsx", type="document", date="2024-01-10",
             summary="2024 annual budget spreadsheet", size=153600),
        File(id="h003", name="team_photo.jpg", type="image", date="2024-02-14",
             summary="team Valentine's Day photo", size=5120000),
        File(id="h004", name="client_contract.pdf", type="document", date="2024-02-20",
             summary="client service contract", size=409600),
        File(id="h005", name="presentation_slides.pptx", type="document", date="2024-03-01",
             summary="Q1 review presentation slides", size=2097152),
        File(id="h006", name="logo_design.png", type="image", date="2024-03-10",
             summary="company logo design", size=1048576),
        # Duplicate pair 1: h007 and h008 share same summary + size
        File(id="h007", name="meeting_notes_march.txt", type="document", date="2024-03-15",
             summary="March all-hands meeting notes", size=16384),
        File(id="h008", name="meeting_notes_march_copy.txt", type="document", date="2024-03-16",
             summary="March all-hands meeting notes", size=16384),
        # Duplicate pair 2: h009 and h010 share same summary + size
        File(id="h009", name="invoice_vendor_a.pdf", type="document", date="2024-03-20",
             summary="vendor A invoice March 2024", size=81920),
        File(id="h010", name="invoice_vendor_a_dup.pdf", type="document", date="2024-03-21",
             summary="vendor A invoice March 2024", size=81920),
        # Additional unique files to reach ≥12
        File(id="h011", name="roadmap_2024.pdf", type="document", date="2024-04-01",
             summary="2024 product roadmap", size=614400),
        File(id="h012", name="onboarding_guide.pdf", type="document", date="2024-04-05",
             summary="new employee onboarding guide", size=512000),
    ]
    hard_folders: dict[str, list[str]] = {
        "root": [f.id for f in hard_files],
    }
    hard_solution = TaskSolution(
        expected_renames={},
        expected_placements={
            "h001": "projects",
            "h002": "finance",
            "h003": "photos",
            "h004": "contracts",
            "h005": "projects",
            "h006": "design",
            "h007": "notes",
            "h011": "projects",
            "h012": "hr",
            "h009": "finance",
        },
        expected_deletions={"h008", "h010"},
    )
    TASKS["hard"] = Task(
        name="hard",
        instruction=(
            "Organize your Downloads folder by creating category folders, moving files into them, "
            "and deleting all duplicate files. "
            "Create folders for 'projects', 'finance', 'photos', 'contracts', 'design', 'notes', and 'hr'. "
            "Move each file to the appropriate folder, then delete any duplicate files "
            "(files that share the same content summary and size as another file)."
        ),
        initial_files=hard_files,
        initial_folders=hard_folders,
        solution=hard_solution,
    )


_register_tasks()
