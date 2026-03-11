with open('doubtbridge.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace: Move similarity calculation BEFORE capability_keywords check
old_code = """        if any(kw in query_lower for kw in capability_keywords):
            print("\\nAnswer:")
            print("You can ask these types of questions:")
            print("-" * 40)
            print("1. About the topic/content - 'what is this lecture about?'")
            print("2. Specific concepts - 'what is cosine similarity?'")
            print("3. Definitions - 'what is the formula of ...'")
            print("4. Explanations - 'explain how ... works'")
            print("5. Examples - 'give an example of ...'")
            print("6. Limitations - 'what are the limitations'")
            print("7. Technologies used - 'what technologies are used'")
            print("8. Headings/Topics - 'what are the headings'")
            print("9. Any question from your lecture notes!")
            print()
            continue


        top_indices = np.argsort(similarities)[-10:][::-1]"""

new_code = """        # Calculate similarity first (before any checks)
        query_embedding = process_query(model, query)
        similarities = cosine_similarity(query_embedding, note_embeddings)[0]
        
        if any(kw in query_lower for kw in capability_keywords):
            print("\\nAnswer:")
            print("You can ask these types of questions:")
            print("-" * 40)
            print("1. About the topic/content - 'what is this lecture about?'")
            print("2. Specific concepts - 'what is cosine similarity?'")
            print("3. Definitions - 'what is the formula of ...'")
            print("4. Explanations - 'explain how ... works'")
            print("5. Examples - 'give an example of ...'")
            print("6. Limitations - 'what are the limitations'")
            print("7. Technologies used - 'what technologies are used'")
            print("8. Headings/Topics - 'what are the headings'")
            print("9. Any question from your lecture notes!")
            print()
            continue

        top_indices = np.argsort(similarities)[-10:][::-1]"""

content = content.replace(old_code, new_code)

# Also fix the app_name issue
content = content.replace('print(f"\\nAnswer: {app_name}\\n")', 'print("\\nAnswer: DoubtBridge\\n")')
content = content.replace('print(f"App: {app_name}")', 'print("App: DoubtBridge")')

with open('doubtbridge.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed successfully!")
