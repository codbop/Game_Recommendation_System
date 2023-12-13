def find_closet_match(test_str, list2check):
scores = {}
for ii in list2check:
    cnt = 0
    if len(test_str)<=len(ii):
        str1, str2 = test_str, ii
    else:
        str1, str2 = ii, test_str
    for jj in range(len(str1)):
        cnt += 1 if str1[jj]==str2[jj] else 0
    scores[ii] = cnt
scores_values        = numpy.array(list(scores.values()))
closest_match_idx    = numpy.argsort(scores_values, axis=0, kind='quicksort')[-1]
closest_match        = numpy.array(list(scores.keys()))[closest_match_idx]
return closest_match, closest_match_idx