# RecSystem_ML

A recommendation system for a post feed based on user behavior: likes, shares, viewing time, and subscriptions.

## Project goal
To build a personalized recommendation feed.

## Team
| Role | Name | GitHub | Telegram |
|------|-----|--------|----------|
| **Team Lead** | Виктория Жиляева | @viktoria_zhilyaeva | @viktoria_zhilyaeva |
| **ML Engineer** | Милана | @imyourmilla | @imyourmilla |
| **Data Engineer** | Оля | @oladyia | @oladyia |
| **Documentation/Tests** | Катя | @litlsun | @litlsun |

## Planned system architecture

```mermaid
flowchart TD
    subgraph INPUT["Input Data"]
        USERS[Users]
        POSTS[Posts]
    end

    USERS --> INTERACTIONS
    POSTS --> INTERACTIONS

    INTERACTIONS[Interactions<br/>likes/reposts/views]

    INTERACTIONS --> FEATURES

    subgraph FEATURES["Feature Engineering"]
        direction LR
        USER_FEAT[User features]
        POST_FEAT[Post features]
    end

    FEATURES --> TRAINING

    subgraph TRAINING["Model Training"]
        direction TB
        
        subgraph CF["Collaborative Filtering"]
            MATRIX[User-item matrix]
            SIM[Similar users]
        end

        subgraph CB["Content Filtering"]
            TAGS[Post tags]
            TEXT[Text]
        end

        HYBRID[Hybrid model<br/>ALS / LightFM]
        
        CF --> HYBRID
        CB --> HYBRID
    end

    TRAINING --> ENGINE

    subgraph ENGINE["Recommendation Engine"]
        SCORE[Relevance prediction]
        TOP[Top-K recommendations]
        SCORE --> TOP
    end

    ENGINE --> FEED[User Feed]

    FEED -.-> |Feedback<br/>likes/skips| INTERACTIONS
```

# Pipline: 
Raw Data → Preprocessing → Engineering → Model → Ranking → Feed

## Repo structure: 

```bash
project/
├── data/
│   ├── raw/
│   │   ├── users.csv
│   │   ├── posts.csv
│   │   └── interactions.csv
│   └── processed/
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       └── recommend.py
│
├── notebooks/
│
├── tests/
│   └── test_data.py
│
├── .gitignore
├── main.py
├── requirements.txt

