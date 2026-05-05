from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "Me encanta entrenar en el gimnasio y mejorar mi físico"
labels = ["deporte", "tecnología", "comida", "negocios"]

result = classifier(text, labels)

print(result)