class ChatService:
    def __init__(self, rag_service, memory_service):
        self.rag = rag_service
        self.memory = memory_service

    def handle_chat(self, user_input: str) -> str:
        self.memory.save("user", user_input)
        response = self.rag.generate_answer(user_input)
        self.memory.save("assistant", response)
        return response
