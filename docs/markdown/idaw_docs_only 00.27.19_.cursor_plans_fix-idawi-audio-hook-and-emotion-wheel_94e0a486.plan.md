---
name: fix-idawi-audio-hook-and-emotion-wheel
overview: Fix syntax error in useTauriAudio.ts and refactor EmotionWheel.tsx to use consistent emotion schema matching the backend.
todos:
  - id: fix-audio-hook
    content: Fix syntax error in `iDAWi/src/hooks/useTauriAudio.ts`
    status: completed
  - id: refactor-emotion-wheel
    content: Refactor `iDAWi/src/components/SideB/EmotionWheel.tsx` to use new schema
    status: completed
  - id: update-music-brain-mock
    content: Update `iDAWi/src/hooks/useMusicBrain.ts` mock data to match new schema
    status: completed
---

I will fix the syntax error in `iDAWi/src/hooks/useTauriAudio.ts` caused by a bad merge or incomplete edit. 

Then, I will refactor `iDAWi/src/components/SideB/EmotionWheel.tsx` and `iDAWi/src/hooks/useMusicBrain.ts` to align the emotion data structure with the Python backend schema (`intent_schema.py`), which uses lowercase keys (e.g., "grief", "joy") instead of the current mismatched capitalized values.

### implementation details
1.  **Fix `iDAWi/src/hooks/useTauriAudio.ts`**: Remove the dangling `});` and ensure `loadTauriInvoke` is called correctly.
2.  **Refactor `iDAWi/src/components/SideB/EmotionWheel.tsx`**:
    *   Remove duplicate code.
    *   Use the new `Emotion` type and `categoryColors` (lowercase keys) defined at the bottom of the file.
    *   Integrate these types into the component logic.
3.  **Update `iDAWi/src/hooks/useMusicBrain.ts`**:
    *   Update `defaultEmotions` to use lowercase categories matching `EmotionWheel`'s new expectation and the backend `intent_schema.py`.
    *   Ensure the mock data structure matches the `Emotion` interface.

### files to be modified
*   `iDAWi/src/hooks/useTauriAudio.ts`
*   `iDAWi/src/components/SideB/EmotionWheel.tsx`
*   `iDAWi/src/hooks/useMusicBrain.ts`