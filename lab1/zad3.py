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

for repo in reposResp:
    language = repo.get("language", "Brak języka") 
    if language in language_counts:
        language_counts[language] += 1
    else:
        language_counts[language] = 1

sorted_data = dict(sorted(language_counts.items(), key=lambda item: item[1], reverse=True))
for language, count in sorted_data.items():
    print(f"{language} -> {count}")


languages = list(language_counts.keys())
counts = list(language_counts.values())

plt.figure(figsize=(9, 6))
plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=140)
plt.axis('equal')

plt.title( userName + "- wykres języków programowania" )

plt.show()
