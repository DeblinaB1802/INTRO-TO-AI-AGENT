from main import StudyBuddy

# Testing utilities
class TestRunner:
    """Basic testing utilities for the chatbot"""
    
    def __init__(self, agent: StudyBuddy):
        self.agent = agent
    
    async def run_basic_tests(self):
        """Run basic functionality tests"""
        test_cases = [
            "What is machine learning?",
            "Solve: 2x + 5 = 15",
            "Create a quiz on Python programming",
            "Help me plan my study schedule"
        ]
        
        print("🧪 Running basic tests...")
        
        for i, test_query in enumerate(test_cases, 1):
            try:
                print(f"Test {i}: {test_query}")
                response = await self.agent.process_query(test_query)
                print(f"✅ Response received: {response[:100]}...")
                print()
            except Exception as e:
                print(f"❌ Test {i} failed: {str(e)}")
        
        print("🧪 Basic tests completed!")