"""Demo data for three users: u_demo (60+ stored memories, a 120-message history, four
bookings), u_asha and u_ravi (fewer of each). u_demo and u_asha share an area and plumbing
preferences, so they compete for the same provider slots.

The memories cover every status the context builder has to tell apart: explicit and inferred,
active and unconfirmed, superseded history, and forgotten tombstones (forgotten through the real
forget path, so their values are scrubbed from the seeded conversation too). Most keys are
irrelevant to booking on purpose: they must stay out of the model's context.
"""

import random
from datetime import datetime, timedelta

from errand.clock import Clock
from errand.contracts import IST, BookingState, Slot
from errand.ids import thread_id_for
from errand.memory.service import MemoryService
from errand.mocks import MockWorld
from errand.mocks.catalog import make_provider_id
from errand.store import FactSource, Store

EXPLICIT, INFERRED = FactSource.EXPLICIT, FactSource.INFERRED

# (key, value, service tag, source, evidence)
FACTS = [
    ("preferred_area", "HSR Layout", None, EXPLICIT, "said so in September"),
    ("preferred_area", "Koramangala", None, EXPLICIT, "moved; corrected it in October"),
    ("earliest_time", "09:00", None, EXPLICIT, "works from home after 9"),
    ("preferred_time_window", "morning", None, EXPLICIT, "prefers mornings"),
    ("earliest_time", "10:00", "electrician", EXPLICIT, "power can only be cut after 10"),
    ("budget_max_inr", 1500, "plumber", EXPLICIT, "said under ₹1,500 for plumbing"),
    ("budget_max_inr", 1200, "ac_service", INFERRED, "two past AC services at about ₹1,100"),
    ("budget_max_inr", 2500, "cleaning", EXPLICIT, "deep cleaning budget"),
    ("budget_max_inr", 1800, "carpenter", EXPLICIT, "door repair budget"),
    ("preferred_time_window", "afternoon", "cleaning", EXPLICIT, "cleaning after lunch"),
    (
        "technician",
        {"provider_id": "prov_electrician_koramangala_3"},
        "electrician",
        INFERRED,
        "your October booking",
    ),
    ("language", "English", None, EXPLICIT, "chat language"),
    ("contact_channel", "WhatsApp", None, EXPLICIT, "prefers WhatsApp updates"),
    ("pets", "a friendly dog; please ring the bell", None, EXPLICIT, "mentioned the dog"),
    ("parking", "visitor parking in basement B2", None, EXPLICIT, "parking note"),
    (
        "access_notes",
        "security calls flat 4B before letting anyone in",
        None,
        EXPLICIT,
        "gate process",
    ),
    ("payment_method", "UPI after the job", None, EXPLICIT, "payment preference"),
    ("floor", "4th floor, lift available", None, EXPLICIT, "building info"),
    ("weekend_ok", True, None, EXPLICIT, "fine with weekend visits"),
    ("notify_before_minutes", 30, None, EXPLICIT, "wants a heads-up"),
    ("household_size", 3, None, INFERRED, "mentioned family of three"),
    ("work_from_home_days", "Mon, Wed, Fri", None, INFERRED, "availability pattern"),
    ("ac_units", "two split ACs, 1.5 ton", "ac_service", EXPLICIT, "AC details"),
    ("ac_brand", "Daikin", "ac_service", EXPLICIT, "AC brand"),
    ("ac_last_service", "June", "ac_service", INFERRED, "booking history"),
    ("geyser", "Racold 25L in the main bathroom", "plumber", EXPLICIT, "geyser details"),
    ("water_pressure", "low on the 4th floor", "plumber", EXPLICIT, "pressure issue"),
    ("kitchen_sink", "leaks under the tap", "plumber", EXPLICIT, "recurring leak"),
    ("wiring_age", "about 15 years", "electrician", EXPLICIT, "old wiring"),
    ("inverter", "Luminous 900VA", "electrician", EXPLICIT, "inverter details"),
    ("fan_count", 5, "electrician", INFERRED, "past job notes"),
    ("door_type", "teak main door", "carpenter", EXPLICIT, "door details"),
    ("wardrobe", "sliding wardrobe in bedroom 2", "carpenter", EXPLICIT, "furniture"),
    ("paint_finish", "matt", "painter", EXPLICIT, "finish preference"),
    ("paint_colour", "off-white", "painter", EXPLICIT, "colour preference"),
    ("rooms_to_paint", "living room only", "painter", INFERRED, "quote request"),
    (
        "pest_history",
        "cockroaches in the kitchen, treated in March",
        "pest_control",
        EXPLICIT,
        "history",
    ),
    ("pest_chemicals", "odourless only, there's a toddler", "pest_control", EXPLICIT, "safety"),
    ("cleaning_supplies", "bring your own supplies", "cleaning", EXPLICIT, "supplies"),
    ("cleaning_rooms", "2 bedrooms, 2 bathrooms", "cleaning", EXPLICIT, "home size"),
    ("sofa_material", "fabric", "cleaning", INFERRED, "past request"),
    ("previous_address", "Indiranagar 12th Main", None, EXPLICIT, "old flat"),
    ("backup_phone", "a second number ending 4410", None, EXPLICIT, "spare contact"),
    ("birthday_month", "August", None, INFERRED, "small talk"),
    ("preferred_brand_tools", "Bosch", None, INFERRED, "small talk"),
    ("noise_sensitive_hours", "after 9 pm", None, EXPLICIT, "neighbours"),
    ("visitor_id_required", True, None, EXPLICIT, "society rule"),
    ("elevator_booking", "needed for large items", None, EXPLICIT, "society rule"),
    ("preferred_payment_split", "no advance payments", None, EXPLICIT, "payment rule"),
    ("review_habit", "writes reviews after jobs", None, INFERRED, "past behaviour"),
    ("emergency_contact", "neighbour in 4A", None, EXPLICIT, "emergencies"),
    ("water_tank", "overhead tank, cleaned yearly", "plumber", EXPLICIT, "tank info"),
    ("switchboard", "modular switches", "electrician", EXPLICIT, "fixtures"),
    ("ac_remote", "remote for bedroom AC is missing", "ac_service", EXPLICIT, "note"),
    ("balcony_door", "sticks in the monsoon", "carpenter", EXPLICIT, "issue"),
    ("ceiling_damp", "patch in the second bathroom", "painter", EXPLICIT, "issue"),
    ("termite_check", "due before Diwali", "pest_control", INFERRED, "seasonal"),
    ("deep_clean_frequency", "every three months", "cleaning", INFERRED, "pattern"),
    ("preferred_day", "Saturday", None, INFERRED, "booking pattern"),
]
FORGOTTEN = (("previous_address", None), ("backup_phone", None))

# Asha lives in Koramangala too, with the same plumbing preferences as the demo user: the two
# compete for the same provider slots. They also both have upcoming cleanings with the same
# provider (a provider cancellation fans out to both).
ASHA_FACTS = [
    ("preferred_area", "Koramangala", None, EXPLICIT, "said so when signing up"),
    ("earliest_time", "09:00", None, EXPLICIT, "school run first"),
    ("preferred_time_window", "morning", None, EXPLICIT, "prefers mornings"),
    ("budget_max_inr", 1500, "plumber", EXPLICIT, "plumbing under ₹1,500"),
    ("usual_service", "plumber", None, EXPLICIT, "an old building, mostly plumbing"),
    ("language", "Hindi", None, EXPLICIT, "chat language"),
    ("pets", "none", None, EXPLICIT, "said so"),
    ("parking", "street parking only", None, EXPLICIT, "building info"),
    ("access_notes", "ring flat 2C", None, EXPLICIT, "gate process"),
    ("contact_channel", "SMS", None, EXPLICIT, "prefers SMS"),
    ("kitchen_sink", "drains slowly", "plumber", EXPLICIT, "recurring issue"),
    ("household_size", 2, None, INFERRED, "mentioned a flatmate"),
]
RAVI_FACTS = [
    ("preferred_area", "HSR Layout", None, EXPLICIT, "said so"),
    ("earliest_time", "08:00", None, EXPLICIT, "early riser"),
    ("preferred_time_window", "evening", None, EXPLICIT, "works days"),
    ("budget_max_inr", 2000, "electrician", EXPLICIT, "electrical work under ₹2,000"),
    ("usual_service", "electrician", None, EXPLICIT, "smart-home tinkering"),
    ("language", "Kannada", None, EXPLICIT, "chat language"),
    ("pets", "two cats", None, EXPLICIT, "please keep the door shut"),
    ("inverter", "Microtek 1100VA", "electrician", EXPLICIT, "inverter details"),
    ("paint_colour", "sage green", "painter", INFERRED, "last painting job"),
]
OTHER_FACTS = {"u_asha": ASHA_FACTS, "u_ravi": RAVI_FACTS}

# The seeded users and their display names, for the UI's user picker.
SEED_USERS = {"u_demo": "Demo user", "u_asha": "Asha", "u_ravi": "Ravi"}

_REQUESTS = [
    (
        "Can you find an electrician for a flickering light?",
        "I found three electricians in Koramangala for Thursday.",
    ),
    (
        "The AC in the bedroom is making noise, book a service",
        "Booked CoolAir for Saturday at 10:00.",
    ),
    (
        "Need a plumber, the kitchen sink is leaking again",
        "Here are plumbers in Koramangala this week.",
    ),
    ("I used to live at Indiranagar 12th Main, do you cover that?", "Yes, Indiranagar is covered."),
    ("Thanks, that worked well", "Glad it helped!"),
    ("Book deep cleaning for the weekend", "Options for home cleaning on Saturday and Sunday."),
    ("Not before 10 for electrical work please", "Noted: electrical visits not before 10:00."),
    ("My backup number is the one ending 4410", "Got it."),
    ("Is there someone who can fix a sticking balcony door?", "I found carpenters in Koramangala."),
    ("What did the AC service cost last time?", "It was ₹1,100."),
]
_OTHER_REQUESTS = {
    "u_asha": [
        ("The kitchen sink drains slowly again", "Here are plumbers in Koramangala tomorrow."),
        ("Book the first one", "Booked FlowRight for Thursday morning."),
        ("Can I get the bathroom tap looked at too?", "Added a note for the plumber."),
        ("Thanks!", "You're welcome."),
    ],
    "u_ravi": [
        ("Smart switch in the hall stopped working", "Here are electricians in HSR Layout."),
        ("Evening please, after 6", "Options for the evening."),
        ("Can someone repaint the balcony wall?", "I found painters in HSR Layout."),
        ("Cool, thanks", "Glad to help."),
    ],
}

# (user, service, area, provider index, days from today, hour)
_BOOKINGS = [
    ("u_demo", "electrician", "Koramangala", 3, -20, 10),
    ("u_demo", "plumber", "Koramangala", 2, -45, 12),
    ("u_demo", "ac_service", "Koramangala", 1, 2, 10),
    ("u_demo", "cleaning", "Koramangala", 1, 5, 14),
    ("u_asha", "plumber", "Koramangala", 1, -30, 10),
    ("u_asha", "ac_service", "Koramangala", 2, 3, 12),
    ("u_asha", "cleaning", "Koramangala", 1, 6, 10),
    ("u_ravi", "electrician", "HSR Layout", 1, 4, 18),
    ("u_ravi", "painter", "HSR Layout", 2, -15, 14),
]


def seed_demo(store: Store, world: MockWorld, clock: Clock) -> list[str]:
    """Seed each demo user once (memories, conversation, bookings); returns the users seeded now,
    leaving any that already have data alone."""
    seeded = []
    for user_id in SEED_USERS:
        if store.memory.status_counts(user_id) or store.messages.count(user_id):
            continue
        _seed_user(store, world, user_id, clock)
        seeded.append(user_id)
    return seeded


def _seed_user(store: Store, world: MockWorld, user_id: str, clock: Clock) -> None:
    memory = MemoryService(store)
    now = clock.now()
    for key, value, tag, source, evidence in FACTS if user_id == "u_demo" else OTHER_FACTS[user_id]:
        if key == "technician":
            value = {**value, "name": world.get_provider(value["provider_id"]).name}
        if source is EXPLICIT:
            memory.remember(user_id, key, value, service=tag, evidence=evidence)
        else:
            memory.infer(user_id, key, value, service=tag, evidence=evidence)

    history = thread_id_for(user_id, "history")
    if user_id == "u_demo":
        rng = random.Random(7)
        for n in range(60):  # 120 messages over the past two months
            user_text, reply = _REQUESTS[n % len(_REQUESTS)]
            moment = now - timedelta(days=60 - n, hours=rng.randint(0, 8))
            store.messages.add(user_id, history, "user", user_text, created_at=moment)
            store.messages.add(
                user_id, history, "assistant", reply, created_at=moment + timedelta(minutes=1)
            )
        for key, tag in FORGOTTEN:  # scrubs the seeded messages that mention them
            memory.forget(user_id, key, service=tag)
    else:
        for n, (user_text, reply) in enumerate(_OTHER_REQUESTS[user_id]):
            moment = now - timedelta(days=20 - 3 * n)
            store.messages.add(user_id, history, "user", user_text, created_at=moment)
            store.messages.add(
                user_id, history, "assistant", reply, created_at=moment + timedelta(minutes=1)
            )

    if not world.list_bookings(user_id).bookings:
        today = now.astimezone(IST).date()
        for owner, service, area, index, days, hour in _BOOKINGS:
            if owner != user_id:
                continue
            start = datetime.combine(
                today + timedelta(days=days), datetime.min.time(), IST
            ).replace(hour=hour)
            world.seed_booking(
                user_id=user_id,
                provider_id=make_provider_id(service, area, index),
                slot=Slot(start=start, end=start + timedelta(hours=2)),
                status=BookingState.CONFIRMED,
            )
