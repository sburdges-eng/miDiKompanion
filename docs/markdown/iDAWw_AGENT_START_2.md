# How to Start the Autonomous Agent

## Step 1: Download These Files
Download all files from this folder and put them in your iDAWi project root:
- `.cursorrules`
- `TODO.md`
- `SESSION_LOG.md`
- `STUCK_LOG.md`

## Step 2: Open Your Project in Cursor
Open your existing iDAWi folder in Cursor (however you normally do it)

## Step 3: Open Composer
Press `Cmd + Shift + I` to open Composer (multi-file edit mode)

## Step 4: Paste This Prompt

---

COPY EVERYTHING BELOW THIS LINE:

---

Read .cursorrules, TODO.md, SESSION_LOG.md, and STUCK_LOG.md to understand your mission.

You are an autonomous agent. Work through TODO.md in priority order:
1. 🔴 CRITICAL first
2. 🟡 TYPESCRIPT ERRORS second
3. 🟢 ESLINT WARNINGS third
4. Continue down the list

RULES:
- Same error 3x = STOP, log to STUCK_LOG.md, move to next task
- Max 5 attempts per problem
- After EACH fix: git add -A && git commit -m "fix: [description] [auto]"
- After every 5 commits: git push origin dev
- Update SESSION_LOG.md as you work
- Mark completed items in TODO.md

START NOW:
1. Run: npm install
2. Run: npm run type-check
3. Fix errors found, one by one
4. Run: npm run lint
5. Fix warnings found
6. Run: npm run build
7. Fix any build errors
8. Update SESSION_LOG.md with summary
9. git push origin dev

Begin working. I'm leaving.

---

## Step 5: Press Enter and Walk Away

The agent will:
- Install dependencies
- Find and fix errors
- Commit each fix
- Push to dev branch
- Log any stuck items

## When You Return

Check:
- `SESSION_LOG.md` - what got done
- `STUCK_LOG.md` - what needs your help
- `git log --oneline -20` - see commits
- `TODO.md` - see progress
