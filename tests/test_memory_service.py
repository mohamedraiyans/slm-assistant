from app.services.memory_service import MemoryService

def test_memory_store_and_retrieve():
    memory = MemoryService()
    memory.save("user", "hello")
    assert len(memory.get_all()) == 1
