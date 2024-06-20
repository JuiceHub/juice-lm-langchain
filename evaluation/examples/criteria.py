from langchain.evaluation import load_evaluator
from model.large_models.language_models import ChatGLM

llm = ChatGLM()

evaluator = load_evaluator("criteria", criteria="conciseness", llm=llm)

eval_result = evaluator.evaluate_strings(
    prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
    input="What's 2+2?",
)
print(eval_result)
