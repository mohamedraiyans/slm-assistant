history = []

def add_message(user, ai):
    history.append({"user": user, "ai": ai})

def get_history():
    return history