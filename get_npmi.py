import requests


def get_url(words):
    base = 'https://palmetto.demos.dice-research.org/service/npmi?words='
    suffix = words.replace(' ', '%20')
    return ''.join([base, suffix])


def get_npmi(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return float(response.text)
    else:
        raise Exception


if __name__ == '__main__':
    string = 'entry output file program build line section printf'
    url = get_url(string)
    score = get_npmi(url)
    print(score)