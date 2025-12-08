"""
Interrogator: Conversational Emotion Exploration System
Builds emotional profile through multi-turn conversation
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import uuid


class InterrogationStage(Enum):
    """Stages of the interrogation process"""
    INITIAL = "initial"
    BASE_EMOTION = "base_emotion"
    INTENSITY = "intensity"
    SPECIFIC_EMOTION = "specific_emotion"
    CONTEXT = "context"
    READY = "ready"


class InterrogationSession:
    """
    Tracks a single interrogation session with conversation history
    and emotional profile building.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.stage = InterrogationStage.INITIAL
        self.conversation_history: List[Dict[str, str]] = []
        self.emotional_profile: Dict[str, Any] = {
            "base_emotion": None,
            "intensity": None,
            "specific_emotion": None,
            "context": [],
            "confidence": 0.0,
        }
        self.turn_count = 0

    def add_user_message(self, message: str):
        """Add user message to conversation history"""
        self.conversation_history.append({"role": "user", "content": message})
        self.turn_count += 1

    def add_system_message(self, message: str):
        """Add system/question message to conversation history"""
        self.conversation_history.append({"role": "system", "content": message})

    def update_emotional_profile(self, updates: Dict[str, Any]):
        """Update emotional profile with new information"""
        self.emotional_profile.update(updates)
        # Recalculate confidence based on completeness
        confidence_score = 0.0
        if self.emotional_profile["base_emotion"]:
            confidence_score += 0.3
        if self.emotional_profile["intensity"]:
            confidence_score += 0.3
        if self.emotional_profile["specific_emotion"]:
            confidence_score += 0.4
        self.emotional_profile["confidence"] = confidence_score

    def is_ready_to_generate(self) -> bool:
        """Check if we have enough information to generate music"""
        return (
            self.emotional_profile["base_emotion"] is not None
            and self.emotional_profile["intensity"] is not None
            and self.emotional_profile["confidence"] >= 0.7
        )

    def build_intent(self) -> Dict[str, Any]:
        """Build emotional intent from session data"""
        return {
            "base_emotion": self.emotional_profile["base_emotion"],
            "intensity": self.emotional_profile["intensity"],
            "specific_emotion": self.emotional_profile["specific_emotion"],
            "context": self.emotional_profile.get("context", []),
        }


# Question templates by stage
QUESTION_TEMPLATES = {
    InterrogationStage.INITIAL: [
        "What are you feeling right now?",
        "How would you describe your current emotional state?",
        "What emotion is most present for you in this moment?",
    ],
    InterrogationStage.BASE_EMOTION: [
        "Can you tell me more about that feeling?",
        "What's the core emotion underneath? (sad, happy, angry, fear, etc.)",
        "If you had to name the main feeling, what would it be?",
    ],
    InterrogationStage.INTENSITY: [
        "How intense is this feeling? (low, moderate, high, intense)",
        "On a scale from subtle to overwhelming, where does this sit?",
        "How strong is this emotion right now?",
    ],
    InterrogationStage.SPECIFIC_EMOTION: [
        "What specific word captures this feeling? (grief, joy, rage, etc.)",
        "Can you be more specific about the emotion?",
        "What's the nuanced version of this feeling?",
    ],
    InterrogationStage.CONTEXT: [
        "What triggered this feeling?",
        "What's the story behind this emotion?",
        "How do you want the music to make you feel?",
    ],
}


def parse_emotion_from_response(message: str) -> Dict[str, Any]:
    """
    Extract emotional information from user's message.
    Uses simple keyword matching - could be enhanced with NLP.
    """
    message_lower = message.lower()
    extracted = {}

    # Base emotions
    base_emotions = {
        "sad": ["sad", "sadness", "down", "depressed", "melancholy"],
        "happy": ["happy", "happiness", "joy", "glad", "cheerful"],
        "angry": ["angry", "anger", "mad", "furious", "rage"],
        "fear": ["fear", "afraid", "anxious", "worried", "scared"],
        "disgust": ["disgust", "disgusted", "repulsed"],
        "surprise": ["surprise", "surprised", "shocked", "amazed"],
        "neutral": ["neutral", "calm", "peaceful", "fine"],
    }

    for base, keywords in base_emotions.items():
        if any(keyword in message_lower for keyword in keywords):
            extracted["base_emotion"] = base
            break

    # Intensity levels
    intensity_keywords = {
        "low": ["low", "subtle", "mild", "slight", "a little"],
        "moderate": ["moderate", "medium", "somewhat", "fairly"],
        "high": ["high", "strong", "very", "quite"],
        "intense": ["intense", "intensely", "extremely", "deeply"],
        "extreme": ["extreme", "extremely", "overwhelming"],
        "overwhelming": ["overwhelming", "overwhelmed", "consuming"],
    }

    for intensity, keywords in intensity_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            extracted["intensity"] = intensity
            break

    # Specific emotions (common ones)
    specific_emotions = [
        "grief",
        "joy",
        "rage",
        "terror",
        "ecstasy",
        "melancholy",
        "anxiety",
        "elation",
        "despair",
        "euphoria",
        "dread",
        "bliss",
    ]

    for specific in specific_emotions:
        if specific in message_lower:
            extracted["specific_emotion"] = specific
            break

    return extracted


def generate_question(session: InterrogationSession) -> str:
    """
    Generate the next question based on session state.
    """
    # If we have base emotion but not intensity, ask about intensity
    if (
        session.emotional_profile["base_emotion"]
        and not session.emotional_profile["intensity"]
    ):
        session.stage = InterrogationStage.INTENSITY
        questions = QUESTION_TEMPLATES[InterrogationStage.INTENSITY]
        return questions[0]

    # If we have base and intensity but not specific, ask for specific
    if (
        session.emotional_profile["base_emotion"]
        and session.emotional_profile["intensity"]
        and not session.emotional_profile["specific_emotion"]
    ):
        session.stage = InterrogationStage.SPECIFIC_EMOTION
        questions = QUESTION_TEMPLATES[InterrogationStage.SPECIFIC_EMOTION]
        return questions[0]

    # If we have everything, ask for context (optional)
    if session.is_ready_to_generate() and session.turn_count < 5:
        session.stage = InterrogationStage.CONTEXT
        questions = QUESTION_TEMPLATES[InterrogationStage.CONTEXT]
        return questions[0]

    # Default: ask about base emotion
    if not session.emotional_profile["base_emotion"]:
        session.stage = InterrogationStage.BASE_EMOTION
        questions = QUESTION_TEMPLATES[InterrogationStage.BASE_EMOTION]
        return questions[0]

    # Fallback
    return "Tell me more about how you're feeling."


def process_interrogation_message(
    session: InterrogationSession, user_message: str
) -> Dict[str, Any]:
    """
    Process a user message in an interrogation session.
    Returns response with question or ready status.
    """
    # Add user message to history
    session.add_user_message(user_message)

    # Extract emotional information
    extracted = parse_emotion_from_response(user_message)

    # Update emotional profile
    if extracted:
        session.update_emotional_profile(extracted)

    # Check if ready to generate
    if session.is_ready_to_generate():
        session.stage = InterrogationStage.READY
        intent = session.build_intent()
        session.add_system_message(
            f"I understand. You're feeling {intent['base_emotion']} "
            f"({intent['intensity']}): {intent['specific_emotion']}. "
            "Ready to generate music!"
        )
        return {
            "ready": True,
            "intent": intent,
            "session_id": session.session_id,
            "message": session.conversation_history[-1]["content"],
            "confidence": session.emotional_profile["confidence"],
        }

    # Generate next question
    question = generate_question(session)
    session.add_system_message(question)

    return {
        "ready": False,
        "question": question,
        "session_id": session.session_id,
        "profile": {
            "base_emotion": session.emotional_profile["base_emotion"],
            "intensity": session.emotional_profile["intensity"],
            "specific_emotion": session.emotional_profile["specific_emotion"],
            "confidence": session.emotional_profile["confidence"],
        },
    }
