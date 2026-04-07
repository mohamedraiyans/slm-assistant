class MemoryService:
    def __init__(self):
        self.history = []

    def save(self, role: str, message: str):
        self.history.append({"role": role, "message": message})

    def get_all(self):
        return self.history
