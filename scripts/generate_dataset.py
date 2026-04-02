import pandas as pd
import random

# Impostiamo un "seme" per rendere i risultati riproducibili ( ogni volta che eseguiamo lo script, otterremo lo stesso risultato )
random.seed(42)

# 1. VOCABOLARIO DI PAROLE PER COSTRUIRE LE RECENSIONI

# Parole tipiche per ogni reparto - POSITIVE
housekeeping_pos = [
    "The room was spotless and very clean.",
    "Excellent housekeeping, everything was tidy.",
    "The bathroom was immaculate, great job.",
    "Fresh towels every day, loved it.",
    "The bed sheets were clean and smelled wonderful.",
    "Room cleaning was done quickly and thoroughly.",
    "The staff cleaned our room perfectly.",
    "Very clean environment, no complaints at all.",
    "The chembermaids did an outstanding job.",
    "Pillows and blankets were fresh and clean."
]

housekeeping_neg = [
    "The room was dirty and not cleaned properly.",
    "Bathroom was disgusting, mold on the walls.",
    "Towels were old and smelled bad.",
    "The floor had dust everywhere, terrible.",
    "Room was never cleaned during our stay.",
    "Sheets looked used and not washed.",
    "Housekeeping forgot to clean our room twice.",
    "Very poor cleaning standards, disappointed.",
    "The toilet was not cleaned at all.",
    "Found hair in the bathroom, unacceptable."
]

reception_pos = [
    "Check-in was fast and the staff was friendly.",
    "The receptionist was very helpful and kind.",
    "Smooth check-out process, no issues at all.",
    "Front desk staff went above and beyond.",
    "Quick and easy check-in, great welcome.",
    "The staff at reception was professional.",
    "Payment was handled quickly and correctly.",
    "Very welcoming reception team, felt at home.",
    "The concierge gave us excellent suggestions.",
    "Check-out was painless and very efficient."
]

reception_neg = [
    "Check-in took forever, very long queue.",
    "The receptionist was rude and unhelpful.",
    "We were charged incorrectly at check-out.",
    "Staff at front desk ignored us completely.",
    "Nobody helped us with our luggage.",
    "Check-in was disorganized and chaotic.",
    "The receptionist made several billing errors.",
    "We waited 40 minutes just to check in.",
    "Front desk staff was unfriendly and cold.",
    "Our room was not ready despite early arrival."
]

fb_pos = [
    "Breakfast was absolutely delicious.",
    "The restaurant menu was varied and tasty.",
    "food quality was excellent, loved every meal.",
    "The buffet had a great selection of dishes.",
    "Dinner at the hotel restaurant was fantastic.",
    "Fresh fruit and pastries every morning.",
    "The bar staff made amazing cocktails.",
    "Great coffee and lovely breakfast atmosphere.",
    "The chef prepared outstanding dishes.",
    "Room service food arrived hot and delicious."
]

fb_neg = [
    "Breakfast was cold and tasteless.",
    "Very limited food options at the buffet.",
    "The restaurant was overpriced for poor quality.",
    "Food took too long to arrive at our table.",
    "The coffee was terrible, tasted like water.",
    "Room service never delivered our order.",
    "The dinner was bland and disappointing.",
    "Found a hair in my soup, disgusting.",
    "Bar was closed when it should have been open.",
    "The food made me feel sick, very bad quality."
]

# 2. TITOLI PER OGNI COMBINAZIONE REPARTO + SENTIMENT

titles = {
    ("Housekeeping", "positive"): [
        "Spotless room!", "Excellent cleaning", "Very clean stay", "Great housekeeping", "Immaculate bathroom"
    ],
    ("Housekeeping", "negative"): [
        "Dirty room", "Poor cleaning", "Terrible hygiene", "Room not cleaned", "Disgusting bathroom"
    ],
    ("Reception", "positive"): [
        "Friendly staff", "Smooth check-in", "Great front desk", "Helpful receptionist", "Excellent service"
    ],
    ("Reception", "negative"): [
        "Rude staff", "Long check-in queue", "Billing error", "Unhelpful reception", "Chaotic check-out"
    ],
    ("F&B", "positive"): [
        "Delicious breakfast", "Great restaurant", "Amazing food", "Lovely buffet", " Excellent dining"
    ], 
    ("F&B", "negative"): [
        "Terrible food", "Cold breakfast", "Poor dining experience", "Overpriced menu", "Disappointing restaurant"
    ]
}

# 3. GENERAZIONE DELLE RECENSIONI

departments = ["Housekeeping", "Reception", "F&B"]
sentiments = ["positive", "negative"]

# Dizionario che collega reparto+sentiment alle frasi giuste
phrases = {
    ("Housekeeping", "positive"): housekeeping_pos,
    ("Housekeeping", "negative"): housekeeping_neg,
    ("Reception", "positive"): reception_pos,
    ("Reception", "negative"): reception_neg,
    ("F&B", "positive"): fb_pos,
    ("F&B", "negative"): fb_neg,
}

# 3B. RECENSIONI AMBUGUE (parlano di più reparti)

ambiguous_reviews = [
    {
        "title": "Mixed experience",
        "body": "The room was dirty but the receptionist was very kind and helpful.",
        "department": "Housekeeping",
        "sentiment": "negative"
    },
    {
        "title": "Good food, bad check-in",
        "body": "Breakfast was delicious but check-in took forever and staff was rude.",
        "department": "F&B",
        "sentiment": "negative"
    },
    {
        "title": "Clean room but bad breakfast",
        "body": "The room was spotless and tidy, however the buffet was cold and tasteless.",
        "department": "Housekeeping",
        "sentiment": "positive"
    },
    {
        "title": "Great stay overall",
        "body": "Check-in was smooth and the room was clean, but dinner was disappointing.",
        "department": "Reception",
        "sentiment": "positive"
    },
    {
        "title": "Terrible experience",
        "body": "Room was not cleaned, food was cold and the front desk ignored us completely.",
        "department": "Housekeeping",
        "sentiment": "negative"
    },
    {
        "title": "Almost perfect",
        "body": "The restaurant was fantastic but we found the bathroom dirty on arrival.",
        "department": "F&B",
        "sentiment": "positive"
    },
    {
        "title": "Confusing stay",
        "body": "Staff at reception was helpful but towels were old and food was average.",
        "department": "Reception",
        "sentiment": "positive"
    },
    {
        "title": "Not what we expected",
        "body": "The breakfast was good but the room smelled bad and check-out was chaotic.",
        "department": "F&B",
        "sentiment": "negative"
    },
    {
        "title": "Decent but improvable",
        "body": "Cleaning was done well but the restaurant was overpriced and slow.",
        "department": "Housekeeping",
        "sentiment": "positive"
    },
    {
        "title": "Up and down",
        "body": "Reception was efficient but the room had dust everywhere and food was cold.",
        "department": "Reception",
        "sentiment": "negative"
    }
]

reviews = []
review_id = 1

# Generiamo 50 recensioni per ogni combinazione (3 reaparti x 2 sentiment = 300 totali)
for department in departments:
    for sentiment in sentiments:
        for _ in range(50):
            title = random.choice(titles[(department, sentiment)])
            body = random.choice(phrases[(department, sentiment)])
            reviews.append({
                "id": review_id,
                "title": title,
                "body": body,
                "department": department,
                "sentiment": sentiment
            })
            review_id += 1 

# Aggiungiamo le recensioni ambigue
for review in ambiguous_reviews:
    review["id"] = review_id
    reviews.append(review)
    review_id += 1

# Mescoliamo le recensioni in modo casuale
random.shuffle(reviews)

# 4. SALVATAGGIO IN UN FILE CSV

df = pd.DataFrame(reviews)
df.to_csv("data/reviews.csv", index=False)

print(f"Dataset generato con successo.")
print(f"Totale recensioni: {len(df)}")
print(f"\nDistribuzione per reparto:")
print(df["department"].value_counts())
print(df["sentiment"].value_counts())
