import requests
import matplotlib.pyplot as plt

# ------------------------ API ------------------------------

githubApi = "https://api.github.com"
userName = "MikiKru"

getUserUri = githubApi + "/users/" + userName
userResp = requests.get(getUserUri).json()

getReposUri = githubApi + "/users/" + userName + "/repos"
reposResp = requests.get(getReposUri).json()

# -------------------------------------------------------------

print('Użytkownik -> ['+ userResp.get('login') +'] używał następujących języków programowania:')

language_counts = {}

# Przeskanuj listę elementów i zlicz wystąpienia języków
for repo in reposResp:
    language = repo.get("language", "Brak języka")  # Pobierz wartość z pola "language"
    if language in language_counts:
        language_counts[language] += 1
    else:
        language_counts[language] = 1

# Wyświetl wyniki zliczania
sorted_data = dict(sorted(language_counts.items(), key=lambda item: item[1], reverse=True))
for language, count in sorted_data.items():
    print(f"{language} -> {count}")


# Przygotowanie danych do wykresu
languages = list(language_counts.keys())
counts = list(language_counts.values())

# Tworzenie wykresu kołowego
plt.figure(figsize=(9, 6))
plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Ustawienie wykresu na okrąg

# Dodanie tytułu
plt.title( userName + "- wykres języków programowania" )

# Wyświetlenie wykresu
plt.show()
