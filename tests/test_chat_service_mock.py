from app.services.chat_service import ChatService


class FakeRAG:
    def generate_answer(self, question: str) -> str:
        return "mocked response"


class FakeMemory:
    def __init__(self):
        self.data = []

    def save(self, role: str, message: str):
        self.data.append((role, message))

    def get_all(self):
        return self.data


def test_chat_with_mock():
    # Arrange (setup fake dependencies)
    rag = FakeRAG()
    memory = FakeMemory()
    chat = ChatService(rag, memory)

    # Act
    result = chat.handle_chat("hi")

    # Assert
    assert result == "mocked response"
    assert len(memory.get_all()) == 2  # user + assistant
    assert memory.get_all()[0][0] == "user"
    assert memory.get_all()[1][0] == "assistant"