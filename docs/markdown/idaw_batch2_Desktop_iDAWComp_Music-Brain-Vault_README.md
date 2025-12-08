# 🎵 Music Brain

**Your Personal Music Production Knowledge Base & AI Assistant**

---

## What Is This?

This is your local, searchable, AI-powered music knowledge system. It contains:

- **Your songs** — notes, stems, mixing decisions, project files
- **Your workflows** — templates, chains, checklists you've developed
- **Your gear** — settings, configurations, manuals
- **Music theory** — vocabulary, concepts, reference materials
- **Sample library** — organized catalog of your sounds
- **AI integration** — chat with your knowledge base locally

---

## Quick Navigation

| Folder | What's Inside |
|--------|---------------|
| [[Songs/]] | Individual song projects with notes |
| [[Workflows/]] | Production templates and checklists |
| [[Gear/]] | Equipment setup and preferences |
| [[Theory/]] | Music theory and audio vocabulary |
| [[Samples-Library/]] | Organized sound catalog |
| [[Projects/]] | Non-song projects (sound design, etc.) |
| [[Templates/]] | All note templates |
| [[AI-System/]] | System docs, guides, and tools |

**Start here:** [[Home]] — Your dashboard with live queries

---

## Getting Started

1. **Explore the folders** — see what's already here
2. **Add a song** — use the template in `Songs/.templates/`
3. **Document a workflow** — capture something you do repeatedly
4. **Ask your AI** — once AnythingLLM is set up, query this vault

---

## Tags System

Use these tags consistently across notes:

### Status
- `#status/idea` — initial concept
- `#status/in-progress` — actively working
- `#status/mixing` — in mixing phase
- `#status/mastering` — in mastering
- `#status/complete` — finished
- `#status/archived` — shelved/old

### Genre/Style
- `#genre/rock`
- `#genre/electronic`
- `#genre/ambient`
- `#genre/folk`
- `#genre/experimental`

### Key
- `#key/C` through `#key/B`
- `#key/Am` through `#key/Gm`

### Tempo Range
- `#tempo/slow` — under 80 BPM
- `#tempo/mid` — 80-120 BPM
- `#tempo/fast` — over 120 BPM

### Instrument Focus
- `#inst/guitar`
- `#inst/vocals`
- `#inst/synth`
- `#inst/drums`
- `#inst/bass`

---

## Dataview Queries

Once you install the Dataview plugin, these queries will work:

### All Songs In Progress
```dataview
TABLE status, key, bpm
FROM "Songs"
WHERE status = "in-progress"
SORT file.mtime DESC
```

### Recent Changes
```dataview
TABLE file.mtime as "Modified"
FROM ""
SORT file.mtime DESC
LIMIT 10
```

---

## Version
- Created: 2025
- System: Obsidian + AnythingLLM + Ollama
- Author: Sean

