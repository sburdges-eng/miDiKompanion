"""
Interrogator Conversational System
Builds emotional profiles through multi-turn conversation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random


class InterrogationStage(Enum):
    """Stages of the interrogation process"""
    OPENING = "opening"
    BASE_EMOTION = "base_emotion"
    INTENSITY = "intensity"
    SPECIFIC_EMOTION = "specific_emotion"
    CONTEXT = "context"
    READY = "ready"


@dataclass
class InterrogationSession:
    """Tracks a conversation session"""
    session_id: str
    stage: InterrogationStage = InterrogationStage.OPENING
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Extracted emotional data
    base_emotion: Optional[str] = None
    intensity: Optional[str] = None
    specific_emotion: Optional[str] = None
    context: Optional[str] = None

    # Confidence scores
    confidence: float = 0.0

    def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        self.conversation_history.append({
            "role": "user",
            "message": message
        })

    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation history"""
        self.conversation_history.append({
            "role": "assistant",
            "message": message
        })

    def is_ready_to_generate(self) -> bool:
        """Check if we have enough information to generate music"""
        return (
            self.base_emotion is not None and
            self.intensity is not None and
            self.specific_emotion is not None and
            self.confidence >= 0.7
        )


# Question templates by stage
QUESTION_TEMPLATES = {
    InterrogationStage.OPENING: [
        "What are you feeling right now?",
        "Tell me about what's on your mind.",
        "What emotion would you like to explore through music?",
        "How are you feeling today?",
    ],
    InterrogationStage.BASE_EMOTION: [
        "Would you say you're feeling sad, happy, angry, fearful, disgusted, surprised, or something else?",
        "What's the main emotion you're experiencing right now?",
        "If you had to pick one word for how you feel, what would it be?",
    ],
    InterrogationStage.INTENSITY: [
        "How intense is this feeling? Is it subtle, mild, moderate, strong, intense, extreme, or overwhelming?",
        "On a scale from subtle to overwhelming, where does this emotion sit?",
        "How deeply are you feeling this right now?",
    ],
    InterrogationStage.SPECIFIC_EMOTION: [
        "Can you be more specific? For example, if you're sad, are you feeling grief, melancholy, heartbreak, or something else?",
        "What specific type of {base_emotion} are you experiencing?",
        "There are many shades of {base_emotion}. Which one resonates most?",
    ],
    InterrogationStage.CONTEXT: [
        "What's the story behind this feeling? What happened or what's happening?",
        "Can you tell me what triggered this emotion?",
        "What's the context for what you're feeling?",
    ],
}


# Emotion keywords for extraction
EMOTION_KEYWORDS = {
    "sad": ["sad", "sadness", "sorrow", "melancholy", "depressed", "down", "blue", "upset"],
    "happy": ["happy", "happiness", "joy", "joyful", "elated", "glad", "cheerful", "pleased"],
    "angry": ["angry", "anger", "rage", "furious", "mad", "irritated", "annoyed", "livid"],
    "fear": ["fear", "fearful", "afraid", "anxious", "worried", "terrified", "scared", "nervous"],
    "disgust": ["disgust", "disgusted", "revulsion", "repulsed", "contempt", "sickened"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed", "startled"],
    "neutral": ["neutral", "calm", "peaceful", "content", "fine", "okay", "normal"],
}

INTENSITY_KEYWORDS = {
    "subtle": ["subtle", "slight", "hint", "trace", "faint"],
    "mild": ["mild", "gentle", "soft", "modest"],
    "moderate": ["moderate", "medium", "reasonable", "modest"],
    "high": ["high", "strong", "considerable", "significant"],
    "intense": ["intense", "powerful", "deep", "profound"],
    "extreme": ["extreme", "overwhelming", "intense", "all-consuming"],
    "overwhelming": ["overwhelming", "all-consuming", "completely", "totally", "utterly"],
}


def parse_user_response(message: str, session: InterrogationSession) -> Dict[str, Any]:
    """
    Extract emotional information from user's message.

    Returns:
        Dictionary with extracted data and confidence
    """
    message_lower = message.lower()
    extracted = {
        "base_emotion": None,
        "intensity": None,
        "specific_emotion": None,
        "confidence": 0.0,
    }

    # Extract base emotion
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                extracted["base_emotion"] = emotion
                extracted["confidence"] += 0.3
                break

    # Extract intensity
    for intensity, keywords in INTENSITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                extracted["intensity"] = intensity
                extracted["confidence"] += 0.2
                break

    # Extract specific emotion (heuristic - look for emotion words)
    specific_emotions = [
        "grief", "bereaved", "mournful", "heartbroken",
        "joy", "elation", "euphoria",
        "rage", "fury", "wrath",
        "terror", "anxiety", "dread",
        "revulsion", "contempt",
        "astonishment", "wonder",
    ]

    for spec in specific_emotions:
        if spec in message_lower:
            extracted["specific_emotion"] = spec
            extracted["confidence"] += 0.2
            break

    return extracted


def generate_question(session: InterrogationSession) -> str:
    """
    Generate the next question based on session state.

    Args:
        session: Current interrogation session

    Returns:
        Next question string
    """
    # Determine next stage
    if session.base_emotion is None:
        session.stage = InterrogationStage.BASE_EMOTION
    elif session.intensity is None:
        session.stage = InterrogationStage.INTENSITY
    elif session.specific_emotion is None:
        session.stage = InterrogationStage.SPECIFIC_EMOTION
    elif session.context is None:
        session.stage = InterrogationStage.CONTEXT
    else:
        session.stage = InterrogationStage.READY

    # Get question template for current stage
    templates = QUESTION_TEMPLATES.get(session.stage, ["Tell me more."])

    # Replace placeholders
    question = random.choice(templates)
    if "{base_emotion}" in question and session.base_emotion:
        question = question.replace("{base_emotion}", session.base_emotion)

    return question


def build_intent_from_session(session: InterrogationSession) -> Dict[str, Any]:
    """
    Build emotional intent from completed session.

    Args:
        session: Completed interrogation session

    Returns:
        Emotional intent dictionary
    """
    if not session.is_ready_to_generate():
        raise ValueError("Session not ready - missing required information")

    intent = {
        "base_emotion": session.base_emotion,
        "intensity": session.intensity,
        "specific_emotion": session.specific_emotion,
    }

    if session.context:
        intent["context"] = session.context

    return intent


class Interrogator:
    """Main interrogator class"""

    def __init__(self):
        self.sessions: Dict[str, InterrogationSession] = {}

    def get_session(self, session_id: str) -> InterrogationSession:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = InterrogationSession(session_id=session_id)
        return self.sessions[session_id]

    def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process a user message and return response.

        Args:
            session_id: Session identifier
            user_message: User's message

        Returns:
            Dictionary with:
            - ready: bool - Whether ready to generate
            - question: str - Next question (if not ready)
            - intent: dict - Emotional intent (if ready)
            - session_id: str
        """
        session = self.get_session(session_id)

        # Add user message to history
        session.add_user_message(user_message)

        # Extract emotional information
        if session.stage != InterrogationStage.OPENING:
            extracted = parse_user_response(user_message, session)

            # Update session with extracted data
            if extracted["base_emotion"]:
                session.base_emotion = extracted["base_emotion"]
            if extracted["intensity"]:
                session.intensity = extracted["intensity"]
            if extracted["specific_emotion"]:
                session.specific_emotion = extracted["specific_emotion"]

            session.confidence += extracted["confidence"]
            session.confidence = min(session.confidence, 1.0)

        # Check if ready to generate
        if session.is_ready_to_generate():
            intent = build_intent_from_session(session)
            session.stage = InterrogationStage.READY
            session.add_assistant_message(
                f"Perfect! I understand you're feeling {session.specific_emotion} "
                f"({session.base_emotion}, {session.intensity} intensity). "
                f"Let me create music that captures this emotion."
            )

            return {
                "ready": True,
                "intent": intent,
                "session_id": session_id,
                "message": session.conversation_history[-1]["message"],
            }

        # Generate next question
        question = generate_question(session)
        session.add_assistant_message(question)

        return {
            "ready": False,
            "question": question,
            "session_id": session_id,
            "stage": session.stage.value,
            "confidence": session.confidence,
        }
