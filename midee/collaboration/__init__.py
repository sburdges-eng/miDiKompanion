"""
Collaboration Features - Real-time collaborative music creation.

Provides:
- WebSocket-based session sharing
- Version control for song intents
- Collaborative editing with conflict resolution
- Comment and annotation system
- User presence and awareness
"""

from midee.collaboration.session import (
    CollaborativeSession,
    SessionParticipant,
    SessionRole,
    create_session,
    join_session,
    leave_session,
)

from midee.collaboration.websocket import (
    CollaborationServer,
    CollaborationClient,
    Message,
    MessageType,
)

from midee.collaboration.version_control import (
    IntentVersionControl,
    IntentVersion,
    IntentDiff,
    create_version,
    get_history,
    restore_version,
)

from midee.collaboration.editing import (
    Operation,
    OperationType,
    CollaborativeDocument,
    apply_operation,
    transform_operations,
)

from midee.collaboration.comments import (
    Comment,
    Annotation,
    CommentThread,
    add_comment,
    resolve_thread,
)

__all__ = [
    # Session
    "CollaborativeSession",
    "SessionParticipant",
    "SessionRole",
    "create_session",
    "join_session",
    "leave_session",
    # WebSocket
    "CollaborationServer",
    "CollaborationClient",
    "Message",
    "MessageType",
    # Version Control
    "IntentVersionControl",
    "IntentVersion",
    "IntentDiff",
    "create_version",
    "get_history",
    "restore_version",
    # Editing
    "Operation",
    "OperationType",
    "CollaborativeDocument",
    "apply_operation",
    "transform_operations",
    # Comments
    "Comment",
    "Annotation",
    "CommentThread",
    "add_comment",
    "resolve_thread",
]
