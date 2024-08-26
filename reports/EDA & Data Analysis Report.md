---
noteId: "36307e205a3c11ef946e63084737c10c"
tags: []

---

# Music and Mental Health: Insights and Analysis

**Author**: Sirine Maaroufi  
**Date**: August 2024

---

1. [Introduction and Objectives](#introduction)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
   - 3.1. [Demographics](#demographics)
   - 3.2. [Music Listening Habits](#music-listening-habits)
   - 3.3. [Musical Background](#musical-background)
   - 3.4. [Music Preferences](#music-preferences)
   - 3.5. [Mental Health Overview](#mental-health-overview)
   - 3.6. [Music Effects](#music-effects)
4. [Correlation Analysis](#correlation-analysis)
   - 4.1. [Music Preferences and Mental Health](#music-preferences-and-mental-health)
   - 4.2. [Preferred BPM and Music Effect](#preferred-bpm-and-music-effect)
   - 4.3. [Listening Hours and Music Effect](#listening-hours-and-music-effect)
   - 4.4. [Musical Background and Music Effect](#musical-background-and-music-effect)
   - 4.5. [Music Effect on Mental Health](#music-effect-on-mental-health)
   - 4.6. [Correlation Matrix](#correlation-matrix)
5. [Conclusion](#conclusion)
6. [Recommendations](#recommendations)

---

## 1. Introduction and Objectives
### Introduction
Music has long been recognized for its potential to impact emotional and psychological well-being. As digital music consumption increases, understanding how music influences mental health is increasingly relevant. This report analyzes data from the Music and Mental Health dataset to explore the relationship between music preferences, listening habits, and mental health conditions.

### Objectives:
The primary objectives of this analysis are to:

- Examine how different demographic factors such as age and musical background influence music preferences and mental health.
- Analyze the relationship between music listening habits, such as hours spent listening and preferred genres, and their effects on mental health.
- Identify patterns and correlations between music preferences and self-reported mental health conditions to provide insights into how music may influence mental well-being.

## 2. Data Overview

The dataset includes information on demographic variables, music preferences, musical background and Engagement with Music, listening habits, and self-reported mental health conditions. \
The dataset contains 728 rows and 31 columns. Seven columns are numerical and the rest are categorical.

## 3. Exploratory Data Analysis

### 3.1. Demographics
---
- **Age Distribution**\
The dataset reveals a predominance of younger respondents, particularly between **14 and 27 years old**. This age group is likely to have distinct music preferences and listening habits compared to older populations. For instance, younger individuals may gravitate towards genres like Rock and Pop due to their popularity and cultural trends within this demographic. Additionally, younger people are more likely to use modern streaming services, which could affect the frequency and context of their music consumption.
<div style="width: 50%; margin: auto;">

![Age Distribution](/reports/Figures/age_distribution.png)
</div>

---

### 3.2. Music Listening Habits
---
- **Streaming Service Usage**\
The analysis reveals that a few major streaming platforms dominate usage: **60%** of respondents use *Spotify*, and **15%** use YouTube Music. **12%** of respondents do not use any streaming service, while other platforms have relatively low usage.
<div style="width: 70%; margin: auto;">

![Streaming Service Usage](/reports/Figures/streaming_service_usage.png)
</div>

- **Hours per Day Listening to Music**\
The hours per day spent listening to music vary among respondents, with most spending between **1 to 3.5 hours**. This distribution suggests that while music is a daily activity for most, the time dedicated to it varies, potentially impacting its effects on mental health.
<div style="width: 65%; margin: auto;">

![Hours per Day](/reports/Figures/hours_per_day.png)
</div>

- **Listening to Music While Working**\
Almost **80%** of respondents listen to music while working, indicating that music is an integral part of their daily routines. This behavior could suggest that music might play a role in influencing productivity and mental focus, which may be worth exploring.
<div style="width: 50%; margin: auto;">

![Listening to Music While Working](/reports/Figures/while_working.png)
</div>

---

### 3.3. Musical Backgroud
---
- **Musicianship**\
Approximately **17%** of respondents are composers, while **30%** identify as instrumentalists.
<div style="width: 70%; margin: auto;">

![Category Distribution](/reports/Figures/musicianship_distribution.png) </div>

- **Engagement with Music**\
Approximately **72%** of respondents enjoy exploring new music, and **55%** like listening to music in foreign languages.
<div style="width: 70%; margin: auto;">

![Engagement with Music](/reports/Figures/music_engagement_distribution.png) </div>

---

### 3.4. Music Preferences
---
- **Favorite Music Genre**\
The visualization reveals that *Rock* is the most popular genre among respondents, with about **25%** preference. Pop follows at **17%**, and Metal at **13%**. Other genres rank below these, with Latin music being the least favored.
![Favorite Music Genre](/reports/Figures/favorite_genre.png)

- **Genre Preference by Age Group**\
The visualization demonstrates that **Rock music** is the most popular genre across all age groups. \
Preferences among younger individuals are diverse, but this diversity tends to decrease with age.
![Favorite Genre by Age Group](/reports/Figures/age_group_genre.png)

- **Preferred Beats Per Minute (BPM)**
The distribution of preferred Beats Per Minute (BPM) shows a clear preference for moderate tempos, with the majority of respondents favoring BPMs between **90 and 150**. This preference aligns with the finding that rock music is the most popular genre among our respondents, as rock often features moderate tempos within this range.
<div style="width: 70%; margin: auto;">

![BPM Distribution](/reports/Figures/bpm_distribution.png) 
</div>

- **Genre Frequency**\
Genres like *pop* and *rock* are frequently listened to, while other genres see less frequent engagement. 
![Genre Frequency](/reports/Figures/genre_frequency.png)

---
### 3.5. Mental Health Overview
---
- **Self-Reported Mental Illnesses**\
The visualization reveals that **60%** of respondents experience *anxiety*, **50%** deal with *depression*, **35%** struggle with *insomnia*, and **20%** are affected by *Obsessive–compulsive disorder (OCD)*.
<div style="width: 70%; margin: auto;">

![Mental Illnesses percentages](/reports/Figures/percentage_mental_illness.png) </div>

The distribution of self-reported mental health conditions reveals the following patterns:
- *Anxiety* is most commonly experienced at levels between 6 and 8 out of 10.
- *Depression* shows a bimodal distribution, with many respondents reporting levels between 6 and 8 out of 10 and 0 and 2 out of 10.
- The majority of respondents do not report significant levels of *OCD or insomnia*.

![Mental Illnesses Distribution](/reports/Figures/mental_illnesses_distribution.png)

---
### 3.6. Music Effects
---
This chart reveals that **74.5%** of respondents reported that music had a positive impact on their well-being, **23.2%** experienced no effect, and **2.3%** felt that music worsened their condition.
<div style="width: 40%; margin: auto;">

![Music Effect](/reports/Figures/music_effects_pie.png)

</div>

## 4. Correlation Analysis

### 4.1. Music Preferences and Mental Health
---
This visualization reveals genre preferences among respondents with different mental health conditions:

- **Anxiety:** The majority of individuals experiencing anxiety prefer Rock and Pop music.
- **Depression:** Those with depression lean towards Rock, Metal, and Pop as their top choices.
- **Insomnia:** Respondents dealing with insomnia tend to favor Rock and Metal.
- **OCD:** Most people with OCD also prefer Rock and Pop music.
<div style="width: 90%; margin: auto;">

![Music Preferences and Mental Health](/reports/Figures/music_prefrences_mental_illness.png)
</div>

---
### 4.2. Preferred BPM and Music Effect
---
The boxplot analysis of the relationship between Beats Per Minute (BPM) and music's effect shows:
- **Improvement:** BPM ranges between 100 and 140.
- **No Effect:** BPM ranges between 110 and 135.
- **Worsening:** BPM ranges between 112 and 130.

All categories exhibit some *outliers* indicating that individual responses to music's tempo can vary. \
The overlap in BPM ranges across different outcomes suggests that while moderate tempos are generally preferred, the impact of music on mental health varies and *is not solely dependent on BPM*.
<div style="width: 70%; margin: auto;">

![Preferred BPM and Music Effect](/reports/Figures/BPM_music_effects.png)\
</div>

---
### 4.3. Listening Hours and Music Effect
---
The dotplot shows that more hours of music listening generally lead to improved well-being. Most responses cluster between 0 and 5 hours, with improvement seen even at higher durations. There are also more instances of no effect at longer listening times, but no worsening is observed with extended listening hours.
![Listening Hours and Music Effect](/reports/Figures/hr_per_day_music_effects.png)

---
### 4.4. Musical Background and Music Effect
---
Analyzing the relationship between being a musician and the effect of music, there is no clear distinction in music's impact between musicians and non-musicians, as the majority in both groups report improvement.
<div style="width: 70%; margin: auto;">

![Musical background and Music Effect](/reports/Figures/instrumentalist_music_effects.png)
![Musical background and Music Effect](/reports/Figures/composer_music_effects.png)

</div>

---
### 4.5. Music effect on Mental health
---
- **Anxiety & Depression:** Moderate to high levels (4-8 for anxiety, 2-7 for depression) show improvement, but higher levels (5-8 for anxiety, 5-10 for depression) may worsen symptoms.
- **OCD & Insomnia:** Lower levels (0-5 for OCD, 1-6 for insomnia) benefit from music, but higher levels (1-5 for OCD, 3-7 for insomnia) may worsen.

The data indicates that music's impact on mental health varies **by condition and severity**, with consistent patterns and no outliers suggesting reliable insights.

![Music Effects on Mental Health](/reports/Figures/mental_illness_music_effects_combined.png)

---
### 4.6. Correlation Matrix
---
Analyzing the correlation matrix reveals that the variable music_effects shows a notable positive correlation with favorite_genre (0.13) and a negative correlation with while_working (-0.16). It also has a slight negative correlation with exploratory (-0.15).

The negative correlation between music_effects and while_working suggests that although music is frequently listened to during work, it may not always positively impact mental health in this context. This implies that music's effect on mental well-being could vary depending on whether it is used for relaxation or work-related tasks. Further investigation into how different types of music or listening settings affect mental health could provide more insights.

Additionally, mental illnesses exhibit strong correlations with each other for example Depression and Anxiety (0.52), Depression and Insomnia (0.37), and Anxiety and OCD (0.35).
These correlations suggest that individuals experiencing one mental health condition are more likely to experience others, highlighting the interrelated nature of these conditions.

![Correlation Matrix](/reports/Figures/correlation_matrix.png)

## 5. Conclusion
The analysis of the Music and Mental Health dataset provides valuable insights into how music preferences, listening habits, and musical background relate to mental well-being. Key findings include:

1. **Demographics and Music Preferences**: Younger respondents (14-27 years) predominantly favor Rock, Pop, and Metal, with Rock being the most popular genre.

2. **Listening Habits and Mental Health**: Most respondents listen to music daily, with 1 to 3.5 hours being the average duration. Music is often part of their work routine, contributing positively to mental health, partcularly to anxiety and depression.

3. **Musical Background and Engagement**: A significant portion of respondents are either musicians or deeply engaged with music, such as exploring new genres or enjoying music in foreign languages. However, the impact of music on mental health does not show a strong distinction between musicians and non-musicians.

4. **Correlation Analysis**: The correlation matrix indicates that music_effects is positively correlated with favorite_genre (0.13) and negatively correlated with while_working (-0.16) and exploratory (-0.15). The strong correlations between mental health conditions suggest that individuals with one condition are more likely to experience others. Besides, the tempo of the music (BPM) and the number of hours spent listening also play a role in influencing mental health, but the effects are not solely dependent on these factors.

5. **Music’s Impact on Mental Health**: Overall, music tends to have a positive effect on well-being for a majority of respondents. However, its impact varies by mental health condition and severity, suggesting that while music generally benefits mental health, the effect can differ based on individual circumstances.

In summary, the relationship between music and mental health is multifaceted, with various factors such as age, genre preference, listening habits, and mental health conditions influencing how individuals respond to music. These findings suggest that personalized music therapy could be an effective approach to enhancing mental well-being, particularly when tailored to the specific needs and preferences of the individual.

## 6. Recommendations:

1. **For Anxiety and Depression:** Individuals experiencing anxiety and depression may benefit from listening to genres such as Rock and Pop, which are preferred by respondents with these conditions. Additionally, moderate BPMs (100-140) appear to be associated with improvements in mental well-being.

2. **For Insomnia and OCD:** While music can be beneficial, its effects on insomnia and OCD may vary. It might be useful to explore specific music types or listening durations that could help alleviate symptoms of these conditions.

3. **General Music Therapy:** Personalized music therapy programs could be developed to address individual needs based on their music preferences and mental health conditions. Tailoring music therapy to match the specific genres and listening habits of individuals may enhance its effectiveness.

