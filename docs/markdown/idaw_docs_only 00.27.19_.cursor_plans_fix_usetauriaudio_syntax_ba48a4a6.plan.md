---
name: Fix useTauriAudio Syntax
overview: Fix the TypeScript syntax error in `iDAWi/src/hooks/useTauriAudio.ts` by removing 3 orphaned lines (31-33) that break the build.
todos:
  - id: fix-orphaned-lines
    content: Remove orphaned lines 31-33 from useTauriAudio.ts
    status: pending
  - id: verify-build
    content: Run TypeScript check to verify fix
    status: pending
---

# Fix useTauriAudio.ts Syntax Error

## Problem
The file `iDAWi/src/hooks/useTauriAudio.ts` has orphaned code at lines 31-33 that breaks TypeScript compilation:

```typescript
loadTauriInvoke(); // Fire and forget; set invoke if/when tauri loads
    // Tauri not available, will use fallbacks   // <-- orphaned
    invoke = null;                                // <-- orphaned  
  });                                             // <-- orphaned
```

These 3 lines appear to be leftover from a previous try-catch or promise chain that was refactored.

## Solution
Remove lines 31-33, keeping only the function call `loadTauriInvoke();`

## File to Edit
`iDAWi/src/hooks/useTauriAudio.ts`

## Verification
Run `npx tsc --noEmit` to confirm no TypeScript errors remain.