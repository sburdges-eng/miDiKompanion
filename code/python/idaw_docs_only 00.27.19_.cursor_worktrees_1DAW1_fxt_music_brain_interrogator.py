"""
Interrogator - Conversational emotion exploration system
Builds emotional profile through multi-turn conversation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
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


@dataclass
class InterrogationSession:
    """Tracks a single interrogation session"""
    session_id: str
    stage: InterrogationStage = InterrogationStage.INITIAL
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Extracted emotional data
    base_emotion: Optional[str] = None
    intensity: Optional[str] = None
    specific_emotion: Optional[str] = None
    context: Optional[str] = None

    # Confidence scores
    confidence: float = 0.0

    def add_user_message(self, message: str):
        """Add user message to conversation history"""
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "stage": self.stage.value
        })

    def add_system_message(self, message: str):
        """Add system/question message to conversation history"""
        self.conversation_history.append({
            "role": "system",
            "message": message,
            "stage": self.stage.value
        })

    def extract_emotion_from_message(self, message: str) -> Dict[str, Any]:
        """Extract emotional information from user message"""
        message_lower = message.lower()
        extracted = {}

        # Base emotions
        base_emotions = {
            "sad": ["sad", "sadness", "unhappy", "down", "depressed"],
            "happy": ["happy", "happiness", "joy", "glad", "cheerful"],
            "angry": ["angry", "anger", "mad", "furious", "rage"],
            "fear": ["fear", "afraid", "scared", "anxious", "worried"],
            "disgust": ["disgust", "disgusted", "repulsed"],
            "surprise": ["surprise", "surprised", "shocked", "amazed"],
        }

        for base, keywords in base_emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                extracted["base_emotion"] = base
                break

        # Intensity keywords
        intensity_keywords = {
            "low": ["slightly", "a little", "somewhat", "mild"],
            "moderate": ["moderately", "fairly", "quite"],
            "high": ["very", "really", "quite", "strongly"],
            "intense": ["intensely", "extremely", "deeply", "profoundly"],
            "extreme": ["extremely", "overwhelmingly", "completely"],
            "overwhelming": ["overwhelming", "consuming", "all-consuming"],
        }

        for intensity, keywords in intensity_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                extracted["intensity"] = intensity
                break

        # Specific emotions (grief, joy, etc.)
        specific_emotions = {
            "grief": ["grief", "grieving", "loss", "mourning"],
            "joy": ["joy", "joyful", "elation", "ecstatic"],
            "melancholy": ["melancholy", "melancholic", "nostalgic"],
            "rage": ["rage", "fury", "wrath"],
            "anxiety": ["anxiety", "anxious", "worried", "nervous"],
        }

        for specific, keywords in specific_emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                extracted["specific_emotion"] = specific
                break

        return extracted

    def update_from_extraction(self, extracted: Dict[str, Any]):
        """Update session with extracted emotional data"""
        if "base_emotion" in extracted:
            self.base_emotion = extracted["base_emotion"]
        if "intensity" in extracted:
            self.intensity = extracted["intensity"]
        if "specific_emotion" in extracted:
            self.specific_emotion = extracted["specific_emotion"]

    def is_ready_to_generate(self) -> bool:
        """Check if we have enough information to generate music"""
        return (
            self.base_emotion is not None and
            self.intensity is not None and
            self.specific_emotion is not None
        )

    def build_intent(self) -> Dict[str, Any]:
        """Build emotional intent from session data"""
        return {
            "base_emotion": self.base_emotion or "neutral",
            "intensity": self.intensity or "moderate",
            "specific_emotion": self.specific_emotion or "calm",
            "context": self.context,
        }


class Interrogator:
    """Main interrogator class for conversational emotion exploration"""

    # Question templates by stage
    QUESTION_TEMPLATES = {
        InterrogationStage.INITIAL: [
            "What are you feeling right now?",
            "How would you describe your emotional state?",
            "What's on your mind emotionally?",
        ],
        InterrogationStage.BASE_EMOTION: [
            "Can you tell me more about that feeling?",
            "What kind of emotion is that?",
            "Is it more sad, happy, angry, or something else?",
        ],
        InterrogationStage.INTENSITY: [
            "How intense is this feeling?",
            "On a scale from subtle to overwhelming, where would you place this?",
            "Is this feeling mild, moderate, or very strong?",
        ],
        InterrogationStage.SPECIFIC_EMOTION: [
            "Can you be more specific about this emotion?",
            "What's the deeper feeling beneath this?",
            "What word best captures this specific emotion?",
        ],
        InterrogationStage.CONTEXT: [
            "What's causing or triggering this feeling?",
            "What situation led to this emotion?",
            "Is there a specific context or memory associated with this?",
        ],
    }

    def __init__(self):
        self.sessions: Dict[str, InterrogationSession] = {}

    def create_session(self) -> str:
        """Create a new interrogation session"""
        session_id = str(uuid.uuid4())
        session = InterrogationSession(session_id=session_id)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[InterrogationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process user message and return response.

        Returns:
            Dict with:
            - ready: bool - whether ready to generate
            - question: str - next question (if not ready)
            - intent: dict - emotional intent (if ready)
            - session_id: str
        """
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session()
            session = self.get_session(session_id)

        # Add user message
        session.add_user_message(user_message)

        # Extract emotional data from message
        extracted = session.extract_emotion_from_message(user_message)
        session.update_from_extraction(extracted)

        # Advance stage based on what we have
        self._advance_stage(session)

        # Check if ready
        if session.is_ready_to_generate():
            session.stage = InterrogationStage.READY
            intent = session.build_intent()
            session.add_system_message("I think I understand. Ready to generate music that captures this emotion.")

            return {
                "ready": True,
                "intent": intent,
                "session_id": session_id,
                "message": "Ready to generate music!",
            }
        else:
            # Generate next question
            question = self._generate_question(session)
            session.add_system_message(question)

            return {
                "ready": False,
                "question": question,
                "session_id": session_id,
                "stage": session.stage.value,
            }

    def _advance_stage(self, session: InterrogationSession):
        """Advance interrogation stage based on collected data"""
        if session.stage == InterrogationStage.INITIAL:
            if session.base_emotion:
                session.stage = InterrogationStage.BASE_EMOTION

        if session.stage == InterrogationStage.BASE_EMOTION:
            if session.intensity:
                session.stage = InterrogationStage.INTENSITY

        if session.stage == InterrogationStage.INTENSITY:
            if session.specific_emotion:
                session.stage = InterrogationStage.SPECIFIC_EMOTION

        if session.stage == InterrogationStage.SPECIFIC_EMOTION:
            if session.context:
                session.stage = InterrogationStage.CONTEXT

    def _generate_question(self, session: InterrogationSession) -> str:
        """Generate appropriate question for current stage"""
        import random

        # Determine what question to ask based on missing data
        if not session.base_emotion:
            templates = self.QUESTION_TEMPLATES[InterrogationStage.BASE_EMOTION]
            return random.choice(templates)

        if not session.intensity:
            templates = self.QUESTION_TEMPLATES[InterrogationStage.INTENSITY]
            return random.choice(templates)

        if not session.specific_emotion:
            templates = self.QUESTION_TEMPLATES[InterrogationStage.SPECIFIC_EMOTION]
            return random.choice(templates)

        # If we have everything, ask about context (optional)
        if not session.context:
            templates = self.QUESTION_TEMPLATES[InterrogationStage.CONTEXT]
            return random.choice(templates)

        # Fallback
        return "Tell me more about how you're feeling."


# Global interrogator instance
_interrogator = Interrogator()


def get_interrogator() -> Interrogator:
    """Get global interrogator instance"""
    return _interrogator
