import google.generativeai as genai
print(sorted([a for a in dir(genai) if not a.startswith('__')]))
print('\ngenai module repr:\n', genai)
