#!/usr/bin/env python3
"""
ML-Based Word Selection for Lyric Generation
============================================
Uses machine learning to select appropriate words for lyrics based on emotion,
context, and prosody requirements.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class WordSelectionModel(nn.Module):
    """
    Neural network for word selection in lyric generation.
    
    Input: Emotion embedding (64-dim) + context (32-dim) + prosody requirements (16-dim)
    Output: Word probability distribution over vocabulary
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        super().__init__()
        
        # Input: 64 (emotion) + 32 (context) + 16 (prosody) = 112
        self.input_size = 112
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Word embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, emotion: torch.Tensor, context: torch.Tensor, prosody: torch.Tensor):
        # Concatenate inputs
        x = torch.cat([emotion, context, prosody], dim=-1)
        
        # Process through network
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits


class ProsodyAnalyzer:
    """Advanced prosody analysis for lyrics."""
    
    def __init__(self):
        self.meter_patterns = {
            'iambic': [0, 1, 0, 1],  # unstressed, stressed
            'trochaic': [1, 0, 1, 0],  # stressed, unstressed
            'anapestic': [0, 0, 1, 0, 0, 1],
            'dactylic': [1, 0, 0, 1, 0, 0]
        }
    
    def analyze_meter(self, stress_pattern: List[int]) -> Tuple[str, float]:
        """Analyze meter pattern and return best match with confidence."""
        best_match = 'iambic'
        best_score = 0.0
        
        for meter_name, pattern in self.meter_patterns.items():
            score = self._match_pattern(stress_pattern, pattern)
            if score > best_score:
                best_score = score
                best_match = meter_name
        
        return best_match, best_score
    
    def _match_pattern(self, stress: List[int], pattern: List[int]) -> float:
        """Calculate pattern match score."""
        if len(stress) == 0:
            return 0.0
        
        matches = 0
        for i, s in enumerate(stress):
            expected = pattern[i % len(pattern)]
            if s == expected or (s > 0 and expected > 0):
                matches += 1
        
        return matches / len(stress)
    
    def validate_line(self, words: List[str], target_syllables: int, 
                     target_stress: Optional[List[int]] = None) -> Tuple[bool, float]:
        """Validate a line against prosody requirements."""
        total_syllables = sum(self._count_syllables(w) for w in words)
        
        syllable_match = 1.0 - abs(total_syllables - target_syllables) / max(target_syllables, 1)
        
        stress_match = 1.0
        if target_stress:
            actual_stress = self._get_stress_pattern(words)
            stress_match = self._match_pattern(actual_stress, target_stress)
        
        overall_score = (syllable_match + stress_match) / 2.0
        return overall_score > 0.7, overall_score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _get_stress_pattern(self, words: List[str]) -> List[int]:
        """Get stress pattern for words (simplified)."""
        # Simplified: assume first syllable is stressed for most words
        pattern = []
        for word in words:
            syllables = self._count_syllables(word)
            if syllables > 0:
                pattern.append(2)  # Primary stress on first syllable
                pattern.extend([0] * (syllables - 1))
        return pattern


class RhymeQualityScorer:
    """Scores rhyme quality between words."""
    
    def __init__(self):
        # Common rhyme patterns
        self.rhyme_types = {
            'perfect': 1.0,
            'slant': 0.7,
            'assonance': 0.5,
            'consonance': 0.3
        }
    
    def score_rhyme(self, word1: str, word2: str) -> Tuple[float, str]:
        """Score rhyme quality between two words."""
        # Extract ending phonemes (simplified)
        end1 = self._get_ending_sound(word1)
        end2 = self._get_ending_sound(word2)
        
        if end1 == end2:
            return 1.0, 'perfect'
        
        # Check for slant rhyme (similar sounds)
        if self._similar_sounds(end1, end2):
            return 0.7, 'slant'
        
        # Check for assonance (vowel match)
        if self._vowel_match(end1, end2):
            return 0.5, 'assonance'
        
        # Check for consonance (consonant match)
        if self._consonant_match(end1, end2):
            return 0.3, 'consonance'
        
        return 0.0, 'none'
    
    def _get_ending_sound(self, word: str) -> str:
        """Extract ending sound (simplified)."""
        word = word.lower()
        # Return last 2-3 characters as approximation
        return word[-3:] if len(word) >= 3 else word
    
    def _similar_sounds(self, s1: str, s2: str) -> bool:
        """Check if sounds are similar."""
        # Simplified similarity check
        return len(set(s1) & set(s2)) >= 2
    
    def _vowel_match(self, s1: str, s2: str) -> bool:
        """Check if vowels match."""
        vowels = "aeiou"
        v1 = [c for c in s1 if c in vowels]
        v2 = [c for c in s2 if c in vowels]
        return len(v1) > 0 and len(v2) > 0 and v1[-1] == v2[-1]
    
    def _consonant_match(self, s1: str, s2: str) -> bool:
        """Check if consonants match."""
        consonants = "bcdfghjklmnpqrstvwxyz"
        c1 = [c for c in s1 if c in consonants]
        c2 = [c for c in s2 if c in consonants]
        return len(c1) > 0 and len(c2) > 0 and c1[-1] == c2[-1]


class MLWordSelector:
    """ML-based word selector for lyric generation."""
    
    def __init__(self, model_path: Optional[Path] = None, vocab_path: Optional[Path] = None):
        self.prosody_analyzer = ProsodyAnalyzer()
        self.rhyme_scorer = RhymeQualityScorer()
        
        # Load vocabulary
        self.vocab = self._load_vocabulary(vocab_path)
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        
        # Initialize model
        self.model = WordSelectionModel(vocab_size=self.vocab_size)
        if model_path and model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
    
    def _load_vocabulary(self, vocab_path: Optional[Path]) -> List[str]:
        """Load vocabulary from file or use default."""
        if vocab_path and vocab_path.exists():
            with open(vocab_path, 'r') as f:
                return json.load(f)
        
        # Default vocabulary (common lyric words)
        return [
            "love", "heart", "soul", "light", "dark", "night", "day", "time",
            "dream", "feel", "know", "see", "come", "go", "take", "make",
            "give", "live", "want", "need", "say", "way", "stay", "play",
            "cry", "fly", "try", "die", "lie", "high", "sky", "eye"
        ] * 100  # Expand for demo
    
    def select_words(
        self,
        emotion: np.ndarray,
        context: np.ndarray,
        prosody_req: Dict,
        num_words: int = 10,
        rhyme_target: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Select words based on emotion, context, and prosody."""
        # Convert to tensors
        emotion_tensor = torch.tensor(emotion, dtype=torch.float32).unsqueeze(0)
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Create prosody vector
        prosody_vec = self._create_prosody_vector(prosody_req)
        prosody_tensor = torch.tensor(prosody_vec, dtype=torch.float32).unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(emotion_tensor, context_tensor, prosody_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Get top candidates
        top_probs, top_indices = torch.topk(probs, num_words * 2, dim=-1)
        
        candidates = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            word = self.idx_to_word[idx.item()]
            score = prob.item()
            
            # Apply rhyme filtering if target provided
            if rhyme_target:
                rhyme_score, rhyme_type = self.rhyme_scorer.score_rhyme(word, rhyme_target)
                score *= (1.0 + rhyme_score * 0.5)  # Boost rhyming words
            
            # Filter by prosody requirements
            if self._meets_prosody(word, prosody_req):
                candidates.append((word, score))
        
        # Sort by score and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:num_words]
    
    def _create_prosody_vector(self, prosody_req: Dict) -> np.ndarray:
        """Create prosody requirement vector."""
        vec = np.zeros(16, dtype=np.float32)
        
        # Syllable count (0-7)
        vec[0] = prosody_req.get('target_syllables', 8) / 16.0
        
        # Stress pattern (8-11)
        stress = prosody_req.get('stress_pattern', [])
        for i, s in enumerate(stress[:4]):
            vec[8 + i] = s / 2.0
        
        # Meter type (12-15)
        meter = prosody_req.get('meter', 'iambic')
        meter_idx = {'iambic': 0, 'trochaic': 1, 'anapestic': 2, 'dactylic': 3}.get(meter, 0)
        vec[12 + meter_idx] = 1.0
        
        return vec
    
    def _meets_prosody(self, word: str, prosody_req: Dict) -> bool:
        """Check if word meets prosody requirements."""
        target_syllables = prosody_req.get('target_syllables', None)
        if target_syllables:
            syllables = self.prosody_analyzer._count_syllables(word)
            if abs(syllables - target_syllables) > 2:
                return False
        
        return True


def main():
    """Example usage."""
    selector = MLWordSelector()
    
    # Example: Select words for a sad, low-energy song
    emotion = np.random.randn(64).astype(np.float32)
    context = np.random.randn(32).astype(np.float32)
    prosody = {
        'target_syllables': 1,
        'stress_pattern': [2, 0],
        'meter': 'iambic'
    }
    
    words = selector.select_words(emotion, context, prosody, num_words=10, rhyme_target="love")
    
    print("Selected words:")
    for word, score in words:
        print(f"  {word}: {score:.4f}")


if __name__ == "__main__":
    main()
