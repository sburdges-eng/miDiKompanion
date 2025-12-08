#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "music_brain/emidi/Event.hpp"
#include "music_brain/emidi/EventConverters.hpp"
#include "music_brain/emidi/NoteEventStub.hpp"

namespace py = pybind11;

PYBIND11_MODULE(emidi_cpp, m) {
    using emidi::Event;

    py::class_<music_brain::NoteEvent>(m, "NoteEvent")
        .def(py::init<>())
        .def_readwrite("pitch", &music_brain::NoteEvent::pitch)
        .def_readwrite("velocity", &music_brain::NoteEvent::velocity)
        .def_readwrite("start_tick", &music_brain::NoteEvent::start_tick)
        .def_readwrite("duration_ticks", &music_brain::NoteEvent::duration_ticks);

    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def_readwrite("note", &Event::note)
        .def_readwrite("velocity", &Event::velocity)
        .def_readwrite("channel", &Event::channel)
        .def_readwrite("start_tick", &Event::start_tick)
        .def_readwrite("duration_ticks", &Event::duration_ticks)
        .def_readwrite("track", &Event::track)
        .def_readwrite("intent_id", &Event::intent_id)
        .def_readwrite("rule_break", &Event::rule_break)
        .def_readwrite("emotional_effect", &Event::emotional_effect)
        .def_readwrite("vulnerability", &Event::vulnerability)
        .def_readwrite("complexity", &Event::complexity)
        .def_readwrite("swing", &Event::swing)
        .def_readwrite("tags", &Event::tags);

    m.def(
        "from_note_event",
        [](const music_brain::NoteEvent& note,
           const std::string& intent_id,
           const std::string& rule_break,
           const std::string& emotional_effect) {
            return emidi::from_note_event(note, intent_id, rule_break, emotional_effect);
        },
        py::arg("note_event"),
        py::arg("intent_id") = "",
        py::arg("rule_break") = "",
        py::arg("emotional_effect") = ""
    );

    m.def(
        "to_note_events",
        [](const std::vector<Event>& events) {
            return emidi::to_note_events(events);
        },
        py::arg("events")
    );
}

