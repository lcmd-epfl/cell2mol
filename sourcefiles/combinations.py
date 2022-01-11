# In[17]:
def subset_sum_iter_mod(array, target):
    sign = 1
    sol_val = []
    sol_idx = []
    #array = sorted(array)  #Sergi
    if target < 0:
        #array = reversed(array)
        sign = -1

    last_index = {0: [-1]}
    for i in range(len(array)):
        for s in list(last_index.keys()):
            new_s = s + array[i]
            if 0 < (new_s - target) * sign:
                pass # Cannot lead to target
            elif new_s in last_index:
                last_index[new_s].append(i)
            else:
                last_index[new_s] = [i]

    # Now yield up the answers.
    def recur (new_target, max_i):
        #indexes = []
        try: 
            for i in last_index[new_target]:
                if i == -1:
                    yield ([],[]) # Empty sum.
                elif max_i <= i:
                    break # Not our solution.
                else:
                    for answer_t in recur(new_target - array[i], i):
                        answer = answer_t[0]
                        idxes = answer_t[1]
                        answer.append(array[i])
                        idxes.append(i)
                        answer_t = (answer,idxes)
                        yield answer_t
        except:
            pass

    for answer_t in recur(target, len(array)):
        sol_val.append(answer_t[0])
        sol_idx.append(answer_t[1])
        
    return sol_val, sol_idx

# In[18]:
def combinations(fglist, reflist, typ="All"):   

    wt = []
    comblist = []
    codelength = len(fglist[0].elemcountvec)
    
    for idx, f in enumerate(fglist):
        if (typ == "Heavy"): 
            wt.append(f.natoms-f.numH)
        elif (typ == "All"):
            wt.append(f.natoms)

    for jdx, ref in enumerate(reflist):
        
        if (typ == "Heavy"): 
            sol_val, sol_idx = subset_sum_iter_mod(wt, ref.natoms-ref.numH)
        elif (typ == "All"):
            sol_val, sol_idx = subset_sum_iter_mod(wt, ref.natoms)
            
        for c in sol_idx:
            tmp = np.zeros((len(fglist)))
            for t in c:  
                ix = int(t)
                tmp[ix] = int(1)
            
            if (np.sum(tmp) > 1): 
                code = np.zeros((codelength))
                for jdx, times in enumerate(tmp):
                    if (times == 1):
                        if (typ == "Heavy"):
                            code = np.add(code, fglist[jdx].Hvcountvec)
                        elif (typ == "All"):
                            code = np.add(code, fglist[jdx].elemcountvec)

                for ref in reflist:
                    if (typ == "Heavy"):
                        if (np.sum(code) == np.sum(ref.Hvcountvec)):
                            if (code == ref.Hvcountvec).all():
                                comblist.append(tmp)
                    elif (typ == "All"):
                        if (np.sum(code) == np.sum(ref.elemcountvec)):
                            if (code == ref.elemcountvec).all():
                                comblist.append(tmp)

    return sorted(comblist, key=sum)
