#use to transform face recognition
#Author Zkx@__@
#Date 2018-01-22

def score_transform(score_level1, score_level2, score_level3, distance):
    if distance < 0.3333 * score_level1:
        return 100
    if distance >= 0.3333 * score_level1 and distance < score_level1:
        score = (1.5*(score_level1 - distance) / score_level1*0.2 + 0.8) * 100
        return score
    if distance >= score_level1 and distance <= score_level2:
        score = (0.8-(distance - score_level1) / (score_level2 - score_level1)*0.2)*100
        return score
    if distance > score_level2 and distance <score_level3:
        score = ((score_level3 - distance) / (score_level3 - score_level2) * 0.6)*100
        return  score
    if distance > score_level3:
        return 0

if __name__ == '__main__':
    score_level1 = 7.07725
    score_level2 = 8.56146
    score_level3 = 17.202581

    distance = [7.7422]

    for elem in distance:
        score = score_transform(score_level1, score_level2, score_level3,elem)
        print elem, score

