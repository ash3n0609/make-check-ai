from firebase_config import db
from firebase_admin import firestore

def save_chat_title(user_id, chat_id, title):
    db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .document(chat_id) \
      .set({
          "title": title,
          "updated_at": firestore.SERVER_TIMESTAMP
      }, merge=True)

def save_message(user_id, chat_id, role, text, metadata=None):
    # Ensure chat exists
    db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .document(chat_id) \
      .set({"updated_at": firestore.SERVER_TIMESTAMP}, merge=True)

    message_data = {
        "role": role,
        "text": text,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    if metadata:
        message_data["metadata"] = metadata

    db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .document(chat_id) \
      .collection("messages") \
      .add(message_data)


def get_user_chats(user_id):
    docs = db.collection("users") \
        .document(user_id) \
        .collection("chats") \
        .order_by("updated_at", direction=firestore.Query.DESCENDING) \
        .stream()
    
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]

def get_chat_history(user_id, chat_id):
    docs = db.collection("users") \
        .document(user_id) \
        .collection("chats") \
        .document(chat_id) \
        .collection("messages") \
        .order_by("timestamp") \
        .stream()

    return [doc.to_dict() for doc in docs]