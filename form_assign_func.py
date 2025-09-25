
# Needed Imports
import pandas as pd
import numpy as np
import math
import re
from tqdm import tqdm


#### Functions related to creating the database of formulas

# Atomic masses - https://ciaaw.org/atomic-masses.htm
# Isotopic abundances - https://ciaaw.org/isotopic-abundances.htm /
# https://www.degruyter.com/view/journals/pac/88/3/article-p293.xml

# Isotopic abundances from  Pure Appl. Chem. 2016; 88(3): 293–306,
# Isotopic compositions of the elements 2013 (IUPAC Technical Report), doi: 10.1515/pac-2015-0503
chemdict = {'H':(1.007825031898, 0.99984426),
            'C':(12.000000000, 0.988922),
            'N':(14.00307400425, 0.996337),
            'O':(15.99491461926, 0.9976206),
            'Na':(22.98976928195, 1.0),
            'P':(30.97376199768, 1.0),
            'S':(31.97207117354, 0.9504074),
            'Cl':(34.968852694, 0.757647),
            'F':(18.99840316207, 1.0),
            'K':(38.96370648482, 0.932581),
            'C13':(13.00335483534, 0.011078),
            'H2': (2.014101777844, 0.00015574),
            'O18':(17.99915961214, 0.0020004),
            'N15':(15.00010889827, 0.003663),
            'S34':(33.967867011, 0.0419599),
            'Cl37': (36.965902573, 0.242353)} 


elem_pattern = r'[A-Z][a-z]?\d*'
elem_groups = r'([A-Z][a-z]?)(\d*)'

def element_composition(formula, elements=None):
    """Returns dictionary of element composition.
    
    Can restrict to given list of elements."""
    
    composition = {}
    for elemp in re.findall(elem_pattern, formula):
        match = re.match(elem_groups, elemp)
        n = match.group(2)
        number = int(n) if n != '' else 1
        composition[match.group(1)] = number
    
    if elements is None:
        return composition
    
    return {e : composition.get(e, 0) for e in elements}


def isotopic_masses(mass):
    "The masses of the most common isotopes for a given mass."
    C_dif = chemdict['C13'][0] - chemdict['C'][0]
    H_dif = chemdict['H2'][0] - chemdict['H'][0]
    O_dif = chemdict['O18'][0] - chemdict['O'][0]
    N_dif = chemdict['N15'][0] - chemdict['N'][0]
    S_dif = chemdict['S34'][0] - chemdict['S'][0]
    Cl_dif = chemdict['Cl37'][0] - chemdict['Cl'][0]
    C2_dif = (chemdict['C13'][0] - chemdict['C'][0]) *2
    H2_dif = (chemdict['H2'][0] - chemdict['H'][0]) *2
    CH_dif = chemdict['C13'][0] - chemdict['C'][0] + chemdict['H2'][0] - chemdict['H'][0]
    
    return {'Monoisotopic': mass, 'C13': mass + C_dif, 'H2': mass + H_dif, 'O18': mass + O_dif, 'N15': mass + N_dif,
            'S34': mass + S_dif, 'Cl37': mass + Cl_dif, '2C13': mass + C2_dif, '2H2': mass + H2_dif, 'C13H2': mass + CH_dif}


def getmass(c,h,o,n,s,p,cl,f):
    "Get the exact mass for any formula."
    massC = chemdict['C'][0] * c
    massH = chemdict['H'][0] * h
    massO = chemdict['O'][0] * o
    massN = chemdict['N'][0] * n
    massS = chemdict['S'][0] * s
    massP = chemdict['P'][0] * p
    massCl = chemdict['Cl'][0] * cl
    massF = chemdict['F'][0] * f 

    massTotal = massC + massH + massO + massN + massS + massP + massCl + massF

    return massTotal


def getabun(c,h,o,n,s,cl):
    "The natural abundance of the formulas made for the most common isotopes of each formula."
    abunC, abunH, abunO, abunN, abunS, abunCl = 1,1,1,1,1,1
    abunC13, abunH2, abunO18, abunN15, abunS34 = False, False, False, False, False
    abunCl37, abun2C13, abun2H2, abunC13H2 = False, False, False, False
    
    if c > 0:
        abunC = chemdict['C'][1] ** c
        abunC13 = (chemdict['C'][1] ** (c-1)) * chemdict['C13'][1] * c
        if c > 1:
            abun2C13 = (chemdict['C'][1] ** (c-2)) * chemdict['C13'][1]**2 * math.comb(c,2)

    if h > 0:
        abunH = chemdict['H'][1] ** h
        abunH2 = (chemdict['H'][1] ** (h-1)) * chemdict['H2'][1] * h
        if h > 1:
            abun2H2 = (chemdict['H'][1] ** (h-2)) * chemdict['H2'][1]**2 * math.comb(h,2)
            
    if o > 0:
        abunO = chemdict['O'][1] ** o
        abunO18 = (chemdict['O'][1] ** (o-1)) * chemdict['O18'][1] * o

    if n > 0:
        abunN = chemdict['N'][1] ** n
        abunN15 = (chemdict['N'][1] ** (n-1)) * chemdict['N15'][1] * n

    if s > 0:
        abunS = chemdict['S'][1] ** s
        abunS34 = (chemdict['S'][1] ** (s-1)) * chemdict['S34'][1] * s
        
    if cl > 0:
        abunCl = chemdict['Cl'][1] ** cl
        abunCl37 = (chemdict['Cl'][1] ** (cl-1)) * chemdict['Cl37'][1] * cl

    abunMono = abunC * abunH * abunO * abunN * abunS * abunCl
    
    abunC13 = abunC13 * abunH * abunO * abunN * abunS * abunCl
    abunH2 = abunC * abunH2 * abunO * abunN * abunS * abunCl
    abunO18 = abunC * abunH * abunO18 * abunN * abunS * abunCl
    abunN15 = abunC * abunH * abunO * abunN15 * abunS * abunCl
    abunS34 = abunC * abunH * abunO * abunN * abunS34 * abunCl
    abunCl37 = abunC * abunH * abunO * abunN * abunS * abunCl37
    abun2C13 = abun2C13 * abunH * abunO * abunN * abunS * abunCl
    abun2H2 = abunC * abun2H2 * abunO * abunN * abunS * abunCl
    if abunC13 and abunH2:
        abunC13H2 = abunC13 * abunH2 * abunO * abunN * abunS * abunCl
    
    return {'Monoisotopic': abunMono, 'C13': abunC13, 'H2': abunH2, 'O18': abunO18, 'N15': abunN15,
            'S34': abunS34, 'Cl37': abunCl37, '2C13': abun2C13, '2H2': abun2H2, 'C13H2': abunC13H2}


def elem_check(HCrange, OCrange, NCrange, PCrange, SCrange, FCrange, ClCrange, e_range):
    "Element Ranges Check"
    if HCrange >= e_range['H/C'][0]:
        if HCrange <= e_range['H/C'][1]:
            if OCrange >= e_range['O/C'][0]:
                if OCrange <= e_range['O/C'][1]:
                    if NCrange >= e_range['N/C'][0]:
                        if NCrange <= e_range['N/C'][1]:
                            if PCrange >= e_range['P/C'][0]:
                                if PCrange <= e_range['P/C'][1]:
                                    if SCrange >= e_range['S/C'][0]:
                                        if SCrange <= e_range['S/C'][1]:
                                            if FCrange >= e_range['F/C'][0]:
                                                if FCrange <= e_range['F/C'][1]:
                                                    if ClCrange >= e_range['Cl/C'][0]:
                                                        if ClCrange <= e_range['Cl/C'][1]:
                                                            return True
    return False


# Common range
com_range = {'H/C':(0.2,3.1),'N/C':(0,1.3),'O/C':(0,1.2),'P/C':(0,0.3),'S/C':(0,0.8)} # 99.7% of all existing formulas
# Extended range
ext_range = {'H/C':(0.1,6),'N/C':(0,4),'O/C':(0,3),'P/C':(0,2),'S/C':(0,3)} # 99.99% of all existing formulas


def form_calc(low, high, elem_range, elem_max):
    """Calculates all formulas possible (according to some stipulations) between a certain mass interval.
    
       low: scalar; lower limit of the molecular mass of the formula.
       high: scalar, upper limit of the molecular mass of the formula.
       elem_range: dictionary.
       elem_max: dictionary.
    
       return: dictionary where keys are exact masses and values are tuples with the overall abundance of the monoisotopic 
    'mass', the number of atoms of each elements (in a specific order), bool value to if the formula follows the valency rules 
    when its elements are in their most common valency, the abundance and mass of the isotope with 1 C(13) atom and a defaulted
    False that is meant to see if the C(13) isotope mass or the monoisotopic mass is being used.
    """
    
    """Following the maximum elements from the recommended in 7 Golden Rules paper."""
    # RULE Nº 1
    if high <= 250:
        maxC,maxH,maxO,maxN = elem_max[250]['C'], elem_max[250]['H'], elem_max[250]['O'], elem_max[250]['N']
        maxS,maxP,maxCl,maxF = elem_max[250]['S'], elem_max[250]['P'], elem_max[250]['Cl'], elem_max[250]['F']
    elif high <= 500:
        maxC,maxH,maxO,maxN = elem_max[500]['C'], elem_max[500]['H'], elem_max[500]['O'], elem_max[500]['N']
        maxS,maxP,maxCl,maxF = elem_max[500]['S'], elem_max[500]['P'], elem_max[500]['Cl'], elem_max[500]['F']
    elif high <= 750:
        maxC,maxH,maxO,maxN = elem_max[750]['C'], elem_max[750]['H'], elem_max[750]['O'], elem_max[750]['N']
        maxS,maxP,maxCl,maxF = elem_max[750]['S'], elem_max[750]['P'], elem_max[750]['Cl'], elem_max[750]['F']
    elif high <= 1000:
        maxC,maxH,maxO,maxN = elem_max[1000]['C'], elem_max[1000]['H'], elem_max[1000]['O'], elem_max[1000]['N']
        maxS,maxP,maxCl,maxF = elem_max[1000]['S'], elem_max[1000]['P'], elem_max[1000]['Cl'], elem_max[1000]['F']
    elif high <= 1250:
        maxC,maxH,maxO,maxN = elem_max[1250]['C'], elem_max[1250]['H'], elem_max[1250]['O'], elem_max[1250]['N']
        maxS,maxP,maxCl,maxF = elem_max[1250]['S'], elem_max[1250]['P'], elem_max[1250]['Cl'], elem_max[1250]['F']
    else:
        maxC,maxH,maxO = elem_max['Higher']['C'], elem_max['Higher']['H'], elem_max['Higher']['O']
        maxN,maxS,maxP = elem_max['Higher']['N'], elem_max['Higher']['S'], elem_max['Higher']['P']
        maxCl,maxF = elem_max['Higher']['Cl'], elem_max['Higher']['F']

    # The maximum possible range to use as a start
    ext_range = elem_range['ext_range']
    
    # + 1 is done since we use range
    maxC = min((int(high) / 12), maxC + 1) # max carbon nº is the smaller of the total mass/12 or predefined maxC
    maxH2 = min((maxC * 4), maxH + 1) # max H nº is the smaller of 4 times the nº of carbons or the predefined max hydrogen nº
    maxO2 = min((int(high) / 16), maxO + 1) # max oxygen nº has to be the smaller of the total mass/16 or predefined maxO
    # Maybe those 3 above on the int part should have a +1 next to them.
    maxN2 = maxN + 1
    maxS2 = maxS + 1
    maxP2 = maxP + 1
    maxF2 = maxF + 1
    maxCl2 = maxCl + 1

    allposs = {} # pd.DataFrame(columns = ['abundance', 'c','h','o','n','s','p'])
    
    # Construction of all possible formulas
    for c in tqdm(range(int(maxC))[1:]): # metabolites contain at least 1 C and 1 H
        
        # This process was done for every element, starting from elements with higher atomic masses to lower masses (faster)
        # See the maximum nº of elements considering the established maximum value and the maximum number according to the
        # element ration with carbon
        maxCl = min(maxCl2, int(c * ext_range['Cl/C'][1]+0.99)) # RULE Nº 5
        for cl in range(maxCl):
            massCl = chemdict['C'][0] * c + chemdict['Cl'][0] * cl # Calculate the mass of the formulas thus far
            if massCl < high: # If it's below the high threshold, continue
                # Repeat for other elements
                maxS = min(maxS2, int(c * ext_range['S/C'][1]+0.99)) # RULE Nº 5
                for s in range(maxS):
                    massS = massCl + chemdict['S'][0] * s
                    if massS < high:
                        maxP = min(maxP2, int(c * ext_range['P/C'][1]+0.99)) # RULE Nº 5
                        for p in range(maxP):
                            massP = massS + chemdict['P'][0] * p
                            if massP < high:
                                maxO = min(maxO2, int(c * ext_range['O/C'][1]+0.99)) # RULE Nº 5
                                minO = 0 # If there are P, there is at least 3 O's.
                                if p > 0:
                                    minO = 3*p
                                    minO = min(minO, int(maxO))
                                for o in range(minO, int(maxO)):
                                    massO = massP + chemdict['O'][0] * o
                                    if massO < high:
                                        maxN = min(maxN2, int(c * ext_range['N/C'][1]+0.99)) # RULE Nº 5
                                        for n in range(maxN):
                                            massN = massO + chemdict['N'][0] * n
                                            if massN < high:
                                                #print(n)
                                                if (n + o) <= 2*c: # Adapted Kujawinski criteria nº1
                                                    NOPS_ratio = NOPS(n,o,p,s)
                                                    if NOPS_ratio: # RULE Nº 6 - element probability check - see function below
                                                        maxF = min(maxF2, int(c * ext_range['F/C'][1]+0.99)) # RULE Nº 5
                                                        for f in range(maxF):
                                                            massF = massN + chemdict['F'][0] * f 
                                                            if massF < high:
                                                                maxH = min(ext_range['H/C'][1] * c + 0.99, maxH2)
                                                                for h in range(int(maxH))[1:]:
                                                                    hcrat = float(h)/float(c)
                                                                    if ext_range['H/C'][0] < hcrat: # RULE Nº 4
                                                                        mass = massF + chemdict['H'][0] * h
                                                                        if h <= (2*c + n + p + 2): # Adapted Kujawinski criteria nº2
                                                                            if low < mass < high:
                                                                                # Rule nº 2 (partially) - Valency check
                                                                                Valency, Valency_normal = Lewis_Senior_rules(c,h,o,n,s,p,cl,f)
                                                                                if Valency:
                                                                                    # If the formula passed all the checks
                                                                                    elem_range_check = False
                                                                                    for m, e_range in elem_range.items():
                                                                                        if mass < m or mass > 1250:
                                                                                            HCrange = h/c
                                                                                            OCrange = o/c
                                                                                            NCrange = n/c
                                                                                            PCrange = p/c
                                                                                            SCrange = s/c
                                                                                            FCrange = f/c
                                                                                            ClCrange = cl/c
                                                                                            if mass > 1250:
                                                                                                elem_range_check=elem_check(HCrange, OCrange, NCrange,
                                                                                                                       PCrange, SCrange, FCrange, ClCrange,
                                                                                                                        elem_range['Higher'])
                                                                                            else:
                                                                                                elem_range_check=elem_check(HCrange, OCrange, NCrange,
                                                                                                                       PCrange, SCrange, FCrange, ClCrange,
                                                                                                                        e_range)
                                                                                            break

                                                                                    if elem_range_check:
                                                                                        # Get the data for the formula and store
                                                                                        abuns = getabun(c,h,o,n,s,cl)
                                                                                        abundance = abuns['Monoisotopic']
                                                                                        #abundanceC13 = abuns['C13']
                                                                                        #massC13 = mass + chemdict['C13'][0] - chemdict['C'][0]
                                                                                        allposs[mass] = (abundance,c,h,o,n,s,p,cl,f,
                                                                                                         Valency_normal)

    return allposs


# Rule nº 6 - HNOPS heuristic probability check
def NOPS (n,o,p,s):
    """Checks if the element counts follow the HNOPS heuristic probablility checks as delineated by the 7 golden rules paper.

       n,o,p,s - integers; number of N, O, P and S atoms respectively in the considered formula.

       returns: bool; True if it fulfills the conditions, False if it doesn't."""

    NOPS_ratio = True
    # Check each of the rules
    if (n > 1) and (o > 1) and (p > 1) and (s > 1): #NOPS
        if (n < 10) and (o < 20) and (p < 4) and (s < 3):
            NOPS_ratio = True 
        else:
            NOPS_ratio = False
    elif (n > 3) and (o > 3) and (p > 3): #NOP
        if (n < 11) and (o < 22) and (p < 6):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (o > 1) and (p > 1) and (s > 1): #OPS
        if (o < 14) and (p < 3) and (s < 3):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (n > 1) and (p > 1) and (s > 1): #PSN
        if (n < 10) and (p < 4) and (s < 3):
            NOPS_ratio = True
        else:
            NOPS_ratio = False
    elif (n > 6) and (o > 6) and (s > 6): #PSN
        if (n < 19) and (o < 14) and (s < 8):
            NOPS_ratio = True
        else:
            NOPS_ratio = False

    return NOPS_ratio


def Lewis_Senior_rules(c,h,o=0,n=0,s=0,p=0,cl=0,f=0):
    """See if the formula follows Lewis' and Senior's rules (considering all max possible valency states for each element
    except Cl).

       c,h,o,n,s,p,cl,f - integers; number of C, H, O, N, S, P, Cl and F atoms respectively in the considered formula.

       returns: (bool, bool); (considering max valency of each element, considering normal valency of each element), True if
    it fulfills the conditions, False if it doesn't."""

    # Normal_Valencies, Max_Valencies (only 1 value when it is the same) - Absolute values of the valencies
    valC = 4
    valH = 1 # Odd
    valO = 2
    valN, max_valN = 3, 5 # Odd
    valS, max_valS = 2, 6 
    valP, max_valP = 3, 5 # Odd
    valCl, max_valCl = 1, 7 # Odd 
    valF = 1 # Odd

    Valency = False
    Valency_normal = False
    # 1st rule - The sum of valences or the total number of atoms having odd valences is even.
    if (h + n + p + cl + f) % 2 == 0:
        # Elements with their max valences
        total_max_v = (valC * c) + (valH * h) + (valO * o) + (max_valN * n) + (max_valS * s) + (max_valP * p) + (
            max_valCl * cl) + (valF * f)

        # 2nd rule - The sum of valences is greater than or equal to twice the maximum valence.
        # Ok, this one sonly eliminates small molecules either way and we are searching for molecules with more than 100 Da.

        # 3rd rule - The sum of valences is greater than or equal to twice the number of atoms minus 1.
        natoms = c + h + o + n + s + p + cl + f
        if total_max_v >= (2*(natoms-1)):
            Valency = True
        
        # If the formula follows the rules with the elements with their Maximum Valencies, see if it follows with normal valencies
        if Valency:
            # Elements with their common valences
            total_v = (valC * c) + (valH * h) + (valO * o) + (valN * n) + (valS * s) + (valP * p) + (valCl * cl) + (valF * f)
            # 3rd rule - The sum of valences is greater than or equal to twice the number of atoms minus 1.
            #natoms = c + h + o + n + s + p + cl + f
            if total_v >= (2*(natoms-1)):
                Valency_normal = True

    return Valency, Valency_normal



#### Functions related to matching to the database of formulas

short_range = {'H/C':(0.5,2.2),'N/C':(0,0.6),'O/C':(0,1.2),'P/C':(0,0.3),'S/C':(0,0.5),'F/C':(0,0.5), 'Cl/C':(0,0.5)}


def short_range_eq_f(mass, sr_eq):
    if sr_eq == None:
        return 0
    elif type(sr_eq) == int:
        return sr_eq
    result = 0
    for i in range(len(sr_eq)):
        result += sr_eq[i]*mass**i
    return result


def form_checker_ratios(data, idx, int_col, threshppm, df, dict_iso={}, isotope_check=True, mass_column='NeutralMass',
                       short_range_eq={'H/C': [None, None], 'O/C': [None, None], 'N/C': [None, None], 'P/C': [None, None],
                                       'S/C': [None, None], 'F/C': [None, None], 'Cl/C': [None, None]}):
    """Assigning formulas to an m/z peak based on the distance of the m/z to the formulas present in a given database.
    
       data: pandas DataFrame; DataFrame with the data that will have the formulas being assigned.
       idx: scalar, string or tuple; index of the current metabolite in which a formula will try to be assigned.
       int_col: str; name of the column where the intensities are in the data DataFrame.
       threshppm: scalar; error threshold for formulae in ppm - i.e. relative error threshold.
       df: pandas DataFrame; dataframe with the formulas that are possible to assign to the m/z peak.
       dict_iso: dictionary; dictionary with information about which masses will be considered C(13) peaks and with which 
    formula.
       isotope_check: bool (default: True); True (isotope check is made, taking into account other peaks in the mass list), 
    False (isotope is not made, formula assignment made independent of other peaks in the mass list).
       mass_column: str (default: 'NeutralMass'); name of the column where the masses are in the data DataFrame.
       short_range_eq: dict; dict with the polynomial coefficients to make the polynomial function (using `short_range_eq_f`)
    of each limit of each elemental ratio used in Check 1 - H/C, O/C, N/C, S/C, P/C. If None, the function returns 0. If an int,
    it returns that int.
       
       returns: tuple with the mass given, the formula assigned (np.nan if no formula could be assigned within the 2 given
    thresholds) and the intensity of the peak."""
    
    # Get information on the peak currently to be assigned
    mass = data.loc[idx, mass_column]
    intensity = data.loc[idx, int_col]

    # Calculate mass difference allowed based on the mass and ppm thresholds given
    mass_dif = mass - (mass*1000000/(1000000 + threshppm))
    # Select the formulas from the database dataframe that are within said mass difference to the mass given
    df2 = df.copy()
    df2 = df2[df2.index<= (mass+mass_dif)]
    df2 = df2[df2.index>= (mass-mass_dif)]
    #print(mass)
    contribution = {'Check 0: Only 1 Formula Candidate': False, 'Check 1: Short Range': False,
                    'Check 2: Normal Valency Check': False, 'Check 3-1: No F and Cl': False,
                    'Check 3-2: Lowest F + Cl': False, 'Check 4: Lowest Heteroatom Count': False,
                    'Check 5: Lowest Mass error': False, 'Previous Isotope Assignment': False}


    # No formula is within the mass interval - no formula assigned
    if len(df2) == 0: 
        return(mass, np.nan, np.nan, dict_iso, 'No Assignment', contribution)

    # Isotope Checker
    if isotope_check == True:
        #print('mass in dict_iso:', mass in dict_iso.keys())
        # If the mass was previously assigned to an isotopic peak, get the isotopic peak from dict_iso and assign it
        if mass in dict_iso.keys():
            if len(dict_iso[mass]) != 0:
                contribution['Previous Isotope Assignment'] = True
                iso, df2, iso_mass = dict_iso[mass]

                formula = formulator(df2['C'].values[0], df2['H'].values[0], df2['O'].values[0], df2['N'].values[0],
                                     df2['S'].values[0], df2['P'].values[0], df2['F'].values[0], df2['Cl'].values[0],
                                     False)
                return(mass,
                       iso + ' iso. - ' + formula,
                       iso_mass,
                       dict_iso,
                      'Previous Isotope Assignment',
                      contribution)

        # Search for possible isotopes for each of the candidate formulas in df2
        # Get the minimum intensity in the dataset
        min_int = data.loc[:, int_col].min()
        min_ratio = (min_int / intensity)*0.8 # See minimum ratio that can be seen as an isotope
        # Put a slightly lower threshold than that to account for intensity variability

        isotopic_peaks = {} # Save isotopic peaks that may be assigned
        poss_isotopic_peaks = {} # Save isotopic ratios that could've theoretically been seen

        # Pass through every formula_candidate and see which isotopic peak candidates each ones have
        for el in df2.index:
            ele = df2.loc[el] # Corresponding row

            # Get expected isotopic abundances ratios compared to monoisotopic peak
            abuns = getabun(ele['C'], ele['H'], ele['O'], ele['N'], ele['S'], ele['Cl'])
            abuns_ratios = pd.Series(abuns) / abuns['Monoisotopic']
            abuns_ratios = abuns_ratios.drop('Monoisotopic')
            abuns_ratios = abuns_ratios[abuns_ratios > min_ratio]

            save_iso_candidates = {} # Store of isotopic peaks found for current formula
            # If there are isotopes that can be found, search for them
            if len(abuns_ratios) > 0:
                # Get their masses
                iso_masses = isotopic_masses(mass)

                poss_isotopic_peaks[el] = list(abuns_ratios.index)
                for iso_type in abuns_ratios.index:
                    # See if their are candidates within those masses calculated
                    iso_candidates = data[data.loc[:, mass_column] <= (iso_masses[iso_type] + mass_dif)]
                    iso_candidates = iso_candidates[iso_candidates.loc[:, mass_column] >= (iso_masses[iso_type] - mass_dif)]
                    # An extra filter to remove possible peaks that have already been assigned as isotopes before
                    # First Come, First Serve type of logic
                    iso_candidates = iso_candidates.loc[[
                            i for i in iso_candidates.index if i not in dict_iso.keys()]]

                    # If there are candidates within the calculated masses, see if their intensity is approximately
                    # what would be expected of their abundance ratios - allowing a 60% variation on the ratio computed
                    # to account for the possible big variability in intensity
                    if len(iso_candidates) > 0:
                        theo_intensity = abuns_ratios[iso_type]*intensity
                        intensity_error_margin = 0.6 * theo_intensity
                        iso_candidates = iso_candidates[
                            iso_candidates.loc[:, int_col] <= (theo_intensity + intensity_error_margin)]
                        iso_candidates = iso_candidates[
                            iso_candidates.loc[:, int_col] >= (theo_intensity - intensity_error_margin)]
                        # theo_relative_intensity = the theoretical relative intensity between isotopic and monoisotopic peaks 
                        # according to natural abundancy.
                        # exp_relative_intensity = the experimental relative intensity between the two peaks
                        iso_candidates['theo_rel_int'] = abuns_ratios[iso_type]

                        if len(iso_candidates) > 0:
                            save_iso_candidates[iso_type] = iso_candidates

            # Save results
            isotopic_peaks[el] = save_iso_candidates

        # S34 Check
        forms_to_remove = []
        forms_to_include = []
        for el in poss_isotopic_peaks:
            if 'S34' in poss_isotopic_peaks[el]:
                if 'S34' not in isotopic_peaks[el]:
                    forms_to_remove.append(el)
                else:
                    forms_to_include.append(el)

        if 0 < len(forms_to_include) < len(poss_isotopic_peaks):
            df2 = df2.loc[forms_to_include]
            isotopic_peaks = {k: v for k,v in isotopic_peaks.items() if k in forms_to_include}
        
        elif len(forms_to_remove) < len(poss_isotopic_peaks):
            df2 = df2.drop(forms_to_remove)

        # Isotope Number Check
        n_isos = {}
        for el in df2.index:
            n_isos[el] = len(isotopic_peaks[el])
        n_isos = pd.Series(n_isos)
        n_isos = n_isos[n_isos == n_isos.max()]
        df2 = df2.loc[n_isos.index]
        isotopic_peaks = {k: isotopic_peaks[k] for k in df2.index}

    
    # Criteria for more than one possiblity: Check number of possible formulas, normal_valency allows formula, check formulas
    # with low amounts of different heteroatoms, check formulas with lowest amounts of heteroatoms, make smallest error check.
    #print(df2)

    # Check nº 1: If more than one formula exists:
    if len(df2) > 1:
        #print(df2)
        short_range = {'H/C': [short_range_eq_f(mass, short_range_eq['H/C'][0]),
                               short_range_eq_f(mass, short_range_eq['H/C'][1])],
                      'O/C': [short_range_eq_f(mass, short_range_eq['O/C'][0]),
                              short_range_eq_f(mass, short_range_eq['O/C'][1])],
                      'N/C': [short_range_eq_f(mass, short_range_eq['N/C'][0]),
                              short_range_eq_f(mass, short_range_eq['N/C'][1])],
                      'S/C': [short_range_eq_f(mass, short_range_eq['S/C'][0]),
                              short_range_eq_f(mass, short_range_eq['S/C'][1])],
                      'P/C': [short_range_eq_f(mass, short_range_eq['P/C'][0]),
                              short_range_eq_f(mass, short_range_eq['P/C'][1])],
                      'F/C': [short_range_eq_f(mass, short_range_eq['F/C'][0]),
                              short_range_eq_f(mass, short_range_eq['F/C'][1])],
                      'Cl/C': [short_range_eq_f(mass, short_range_eq['Cl/C'][0]),
                              short_range_eq_f(mass, short_range_eq['Cl/C'][1])]}

        df_range = df2[(df2['H']/df2['C']) <= short_range['H/C'][1]]
        df_range = df_range[(df_range['H']/df_range['C']) >= short_range['H/C'][0]]
        df_range = df_range[(df_range['O']/df_range['C']) >= short_range['O/C'][0]]
        df_range = df_range[(df_range['O']/df_range['C']) <= short_range['O/C'][1]]
        df_range = df_range[(df_range['N']/df_range['C']) >= short_range['N/C'][0]]
        df_range = df_range[(df_range['N']/df_range['C']) <= short_range['N/C'][1]]
        df_range = df_range[(df_range['S']/df_range['C']) >= short_range['S/C'][0]]
        df_range = df_range[(df_range['S']/df_range['C']) <= short_range['S/C'][1]]
        df_range = df_range[(df_range['P']/df_range['C']) >= short_range['P/C'][0]]
        df_range = df_range[(df_range['P']/df_range['C']) <= short_range['P/C'][1]]
        df_range = df_range[(df_range['F']/df_range['C']) >= short_range['F/C'][0]]
        df_range = df_range[(df_range['F']/df_range['C']) <= short_range['F/C'][1]]
        df_range = df_range[(df_range['Cl']/df_range['C']) >= short_range['Cl/C'][0]]
        df_range = df_range[(df_range['Cl']/df_range['C']) <= short_range['Cl/C'][1]]
        if len(df_range) != 0:
            if len(df_range) != len(df2):
                contribution['Check 1: Short Range'] = True
            df2 = df_range
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      dict_iso,
                      'Check 1: Short Range',
                      contribution)#,
        #print(df2_range)
            
    if len(df2) > 1:   
        # Check nº 2: if there are formulas that can exist with elements in their most common valency
        # Con: Formulas with groups NO2 where N has a valency of 4 will be excluded
        df_valency = df2[df2['Valency'] == True]
        if len(df_valency) != 0:
            if len(df_valency) != len(df2):
                contribution['Check 2: Normal Valency Check'] = True
            df2 = df_valency
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      dict_iso,
                      'Check 2: Normal Valency Check',
                      contribution)#,

        # Check nº 3 skipped

        # Check nº 3 part 2: if there are formulas with low amounts of different heteroatoms: prioritize C,H,O,N,S,P.
        df_CHONSP = df2[df2['F'] + df2['Cl'] == 0]
        if len(df_CHONSP) != 0:
            if len(df_CHONSP) != len(df2):
                contribution['Check 3-1: No F and Cl'] = True
            df2 = df_CHONSP
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      dict_iso,
                      'Check 3-1: No F and Cl',
                      contribution)#,
        else:
            df_CHONSP = df2[df2['F'] + df2['Cl'] == (df2['F'] + df2['Cl']).min()]
            if len(df_CHONSP) != 0:
                if len(df_CHONSP) != len(df2):
                    contribution['Check 3-2: Lowest F + Cl'] = True
                df2 = df_CHONSP
                if len(df2) == 1:
                    if isotope_check == True:
                        dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                    return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                            df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                            False),
                        df2.index[0],
                        dict_iso,
                        'Check 3-2: Lowest F + Cl',
                      contribution)#,

        if len(df2) > 1:
            # Check nº 4: Lowest Heteroatom count
            df3 = df2[df2[['N','S','P','F','Cl']].sum(axis = 1) ==
                      df2[['N','S','P','F','Cl']].sum(axis = 1).min()]
            if len(df3) != len(df2):
                contribution['Check 4: Lowest Heteroatom Count'] = True
            df2 = df3
            #df2 = df2[df2[['S','P']].sum(axis = 1) == 
            #          df2[['S','P']].sum(axis = 1).min()]
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      dict_iso,
                      'Check 4: Lowest Heteroatom Count',
                      contribution)#,
                                        #False))

            # Final Check: Lowest error
            # Calculate and store the error (in ppm) of the mass of the filtered formulas
            df3 = pd.DataFrame(abs(((mass - df2.index)/df2.index)*1000000).values, index = df2.index, columns = ['error'])
            mini = df3.idxmin(axis = 'index')
            contribution['Check 5: Lowest Mass error'] = True
            # Choose the formula with the lowest error (closest to original mass).
            if isotope_check == True:
                dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

            return(mass, formulator(df2.loc[mini, 'C'].values[0],df2.loc[mini,'H'].values[0],df2.loc[mini,'O'].values[0],
                                    df2.loc[mini,'N'].values[0],df2.loc[mini,'S'].values[0],df2.loc[mini,'P'].values[0],
                                    df2.loc[mini,'F'].values[0], df2.loc[mini,'Cl'].values[0],
                                    False),
                   mini.iloc[0],
                   dict_iso,
                  'Check 5: Lowest Mass error',
                    contribution)
    
    # Only one formula is within the mass interval - it will be assigned
    elif len(df2) == 1:
        contribution['Check 0: Only 1 Formula Candidate'] = True
        if isotope_check == True:
            dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

        return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],df2['S'].values[0],
                                df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0], False),
              df2.index[0],
              dict_iso,
              'Check 0: Only 1 Formula Candidate',
               contribution)


def isotope_decider(dict_iso, isotopic_peaks, df2, mass, monoiso_int, int_col, mass_column):
    """Decides which isotope candidate is selected as the best candidate for the formula chosen present in df2."""

    # Calculate expected isotopic masses that may be helpful
    iso_masses = isotopic_masses(mass)
    isotopic_peaks = isotopic_peaks[df2.index[0]]

    # Go through every isotope
    for iso in isotopic_peaks:

        iso_candidates = isotopic_peaks[iso]
        # If only 1 candidate
        if len(iso_candidates) == 1:
            dict_iso[iso_candidates[mass_column].iloc[0]] = [iso, df2, iso_masses[iso]]
            #print(iso_candidates)

        elif len(iso_candidates) > 1:
            # 1st criteria - Ratios between monoisotopic and isotopic peak abundances are close (less than 20%) to what is expected
            theo_intensity =  iso_candidates['theo_rel_int'].values[0] * monoiso_int
            intensity_error_margin = 0.2 * theo_intensity
            close_iso_candidates = iso_candidates[
                iso_candidates.loc[:,int_col] <= (theo_intensity + intensity_error_margin)]
            close_iso_candidates = close_iso_candidates[
                close_iso_candidates.loc[:,int_col] >= (theo_intensity - intensity_error_margin)]

            if len(close_iso_candidates) == 1: # If only one
                dict_iso[close_iso_candidates[mass_column].iloc[0]] = [iso, df2, iso_masses[iso]]

            else: # If zero
                if len(close_iso_candidates) > 1: # If more than one
                    iso_candidates = close_iso_candidates
                    
                # 2nd criteria - Closer distance between experimental to the theoretical mass for the isotope candidate
                iso_idx = abs((iso_masses[iso] - iso_candidates[mass_column])/iso_candidates[mass_column]*
                                1000000).astype(float).idxmin()
                dict_iso[iso_candidates[mass_column].loc[iso_idx]] = [iso, df2, iso_masses[iso]]
    return dict_iso


# Adapted from FTMSVisualization
def formulator(c,h,o,n,s,p,f=0,cl=0,c13=0):
    """Transforms element counts to a readable formula in string format. Element order: C, H, N, O, S, P, F, Cl (and C(13))."""
    
    formula = "C"+str(c)+"H"+str(h)

    if cl > 0:
        if cl > 1:
            formula = formula + "Cl" + str(cl)
        else:
            formula = formula + "Cl"
    if f > 0:
        if f > 1:
            formula = formula + "F" + str(f)
        else:
            formula = formula + "F"
    if n > 0:
        if n > 1:
            formula = formula + "N" + str(n)
        else:
            formula = formula + "N"
    if o > 0:
        if o > 1:
            formula = formula + "O" + str(o)
        else:
            formula = formula + "O"
    if p > 0:
        if p > 1:
            formula = formula + "P" + str(p)
        else:
            formula = formula + "P"
    if s > 0:
        if s > 1:
            formula = formula + "S" + str(s)
        else:
            formula = formula + "S"
    if c13 > 0:
        if c13 > 1:
            formula = formula + "C(13)" + str(s)
        else:
            formula = formula + "C(13)"

    return formula


def form_checker_ratios_adducts(data, idx, int_col, threshppm, df, dict_iso={}, isotope_check=True, mass_column='NeutralMass',
                        adducts_to_consider={'M':0}, deviation_in_ppm=True,
                       short_range_eq={'H/C': [None, None], 'O/C': [None, None], 'N/C': [None, None], 'P/C': [None, None], 'S/C': [None, None]}):
    """Assigning formulas to an m/z peak based on the distance of the m/z to the formulas present in a given database.
    
       data: pandas DataFrame; DataFrame with the data that will have the formulas being assigned.
       idx: scalar, string or tuple; index of the current metabolite in which a formula will try to be assigned.
       int_col: str; name of the column where the intensities are in the data DataFrame.
       threshppm: scalar; error threshold for formulae in ppm - i.e. relative error threshold. If deviation_in_ppm is False,
    it is the flat Da deviation allowed
       df: pandas DataFrame; dataframe with the formulas that are possible to assign to the m/z peak.
       dict_iso: dictionary; dictionary with information about which masses will be considered C(13) peaks and with which 
    formula.
       isotope_check: bool (default: True); True (isotope check is made, taking into account other peaks in the mass list), 
    False (isotope is not made, formula assignment made independent of other peaks in the mass list).
       mass_column: str (default: 'NeutralMass'); name of the column where the masses are in the data DataFrame.
       adducts_to_consider: dict (default:{'M':0}); dict with the keys being the name of the adduct to consider for formula
    assignment and the values the corresponding mass shift they cause in the formula mass.
       deviation_in_ppm: bool (default=True); if True, threshppm is taken as ppm deviation, if False, it is taken as flat Da
    deviation.
       short_range_eq: dict; dict with the polynomial coefficients to make the polynomial function (using `short_range_eq_f`)
    of each limit of each elemental ratio used in Check 1 - H/C, O/C, N/C, S/C, P/C. If None, the function returns 0. If an int,
    it returns that int.
       
       returns: tuple with the mass given, the formula assigned (np.nan if no formula could be assigned within the 2 given
    thresholds) and the intensity of the peak."""
    
    # Get information on the peak currently to be assigned
    mass = data.loc[idx, mass_column]
    intensity = data.loc[idx, int_col]

    # Select the formulas from the database dataframe that are within the ppm threshold for the different adducts to the mass given
    joined_df = pd.DataFrame()
    for adduct, mass_dev in adducts_to_consider.items():
        neutral_mass = float(mass - mass_dev)
        # Calculate mass difference allowed based on the mass and ppm thresholds given
        if deviation_in_ppm:
            mass_dif = neutral_mass - (neutral_mass*1000000/(1000000 + threshppm))
        else:
            mass_dif = threshppm
        # Select the formulas from the database dataframe that are within said mass difference to the mass given
        df2 = df.copy()
        df2 = df2[df2.index<= (neutral_mass+mass_dif)]
        df2 = df2[df2.index>= (neutral_mass-mass_dif)]
        df2['Adduct'] = adduct
        df2['Ad. Neutral Mass'] = neutral_mass
        df2['Mass dif.'] = mass_dif
        # Add current adduct result to final df
        joined_df = pd.concat((joined_df, df2))
    # DataFrame with all possible candidates
    df2 = joined_df

    contribution = {'Check 0: Only 1 Formula Candidate': False, 'Check 1: Short Range': False,
                    'Check 2: Normal Valency Check': False, 'Check 3-1: No F and Cl': False,
                    'Check 3-2: Lowest F + Cl': False, 'Check 4: Lowest Heteroatom Count': False,
                    'Check 5: Lowest Mass error': False, 'Previous Isotope Assignment': False}

    # No formula is within the mass interval - no formula assigned
    if len(df2) == 0: 
        return(mass, np.nan, np.nan, np.nan, dict_iso, 'No Assignment', contribution)

    # Isotope Checker
    if isotope_check == True:
        #print('mass in dict_iso:', mass in dict_iso.keys())
        # If the mass was previously assigned to an isotopic peak, get the isotopic peak from dict_iso and assign it
        if mass in dict_iso.keys():
            if len(dict_iso[mass]) != 0:
                contribution['Previous Isotope Assignment'] = True
                iso, df2, iso_mass = dict_iso[mass]

                formula = formulator(df2['C'].values[0], df2['H'].values[0], df2['O'].values[0], df2['N'].values[0],
                                     df2['S'].values[0], df2['P'].values[0], df2['F'].values[0], df2['Cl'].values[0],
                                     False)
                return(mass,
                       iso + ' iso. - ' + formula,
                       iso_mass,
                       df2.iloc[0].loc['Adduct'],
                       dict_iso,
                      'Previous Isotope Assignment',
                      contribution)

        # Search for possible isotopes for each of the candidate formulas in df2
        # Get the minimum intensity in the dataset
        min_int = data.loc[:, int_col].min()
        min_ratio = (min_int / intensity)*0.8 # See minimum ratio that can be seen as an isotope
        # Put a slightly lower threshold than that to account for intensity variability

        isotopic_peaks = {} # Save isotopic peaks that may be assigned
        poss_isotopic_peaks = {} # Save isotopic ratios that could've theoretically been seen

        # Pass through every formula_candidate and see which isotopic peak candidates each ones have
        for el in df2.index:
            ele = df2.loc[el] # Corresponding row

            # Get expected isotopic abundances ratios compared to monoisotopic peak
            abuns = getabun(ele['C'], ele['H'], ele['O'], ele['N'], ele['S'], ele['Cl'])
            abuns_ratios = pd.Series(abuns) / abuns['Monoisotopic']
            abuns_ratios = abuns_ratios.drop('Monoisotopic')
            abuns_ratios = abuns_ratios[abuns_ratios > min_ratio]

            save_iso_candidates = {} # Store of isotopic peaks found for current formula
            # If there are isotopes that can be found, search for them
            if len(abuns_ratios) > 0:
                # Get their masses
                iso_masses = isotopic_masses(mass)
                # Get maximum possible mass_dif from the adduct
                mass_dif = df2.loc[el, 'Mass dif.']

                poss_isotopic_peaks[el] = list(abuns_ratios.index)
                for iso_type in abuns_ratios.index:
                    # See if their are candidates within those masses calculated
                    iso_candidates = data[data.loc[:, mass_column] <= (iso_masses[iso_type] + mass_dif)]
                    iso_candidates = iso_candidates[iso_candidates.loc[:, mass_column] >= (iso_masses[iso_type] - mass_dif)]
                    # An extra filter to remove possible peaks that have already been assigned as isotopes before
                    # First Come, First Serve type of logic
                    iso_candidates = iso_candidates.loc[[
                            i for i in iso_candidates.index if i not in dict_iso.keys()]]

                    # If there are candidates within the calculated masses, see if their intensity is approximately
                    # what would be expected of their abundance ratios - allowing a 60% variation on the ratio computed
                    # to account for the possible big variability in intensity
                    if len(iso_candidates) > 0:
                        theo_intensity = abuns_ratios[iso_type]*intensity
                        intensity_error_margin = 0.6 * theo_intensity
                        iso_candidates = iso_candidates[
                            iso_candidates.loc[:, int_col] <= (theo_intensity + intensity_error_margin)]
                        iso_candidates = iso_candidates[
                            iso_candidates.loc[:, int_col] >= (theo_intensity - intensity_error_margin)]
                        # theo_relative_intensity = the theoretical relative intensity between isotopic and monoisotopic peaks 
                        # according to natural abundancy.
                        # exp_relative_intensity = the experimental relative intensity between the two peaks
                        iso_candidates['theo_rel_int'] = abuns_ratios[iso_type]

                        if len(iso_candidates) > 0:
                            save_iso_candidates[iso_type] = iso_candidates

            # Save results
            isotopic_peaks[el] = save_iso_candidates

        # S34 Check
        forms_to_remove = []
        forms_to_include = []
        for el in poss_isotopic_peaks:
            if 'S34' in poss_isotopic_peaks[el]:
                if 'S34' not in isotopic_peaks[el]:
                    forms_to_remove.append(el)
                else:
                    forms_to_include.append(el)

        if 0 < len(forms_to_include) < len(poss_isotopic_peaks):
            df2 = df2.loc[forms_to_include]
            isotopic_peaks = {k: v for k,v in isotopic_peaks.items() if k in forms_to_include}
        
        elif len(forms_to_remove) < len(poss_isotopic_peaks):
            df2 = df2.drop(forms_to_remove)

        # Isotope Number Check
        n_isos = {}
        for el in df2.index:
            n_isos[el] = len(isotopic_peaks[el])
        n_isos = pd.Series(n_isos)
        n_isos = n_isos[n_isos == n_isos.max()]
        df2 = df2.loc[n_isos.index]
        isotopic_peaks = {k: isotopic_peaks[k] for k in df2.index}

    
    # Criteria for more than one possiblity: Check number of possible formulas, normal_valency allows formula, check formulas
    # with low amounts of different heteroatoms, check formulas with lowest amounts of heteroatoms, make smallest error check.
    #print(df2)

    # Check nº 1: If more than one formula exists:
    if len(df2) > 1:
        # Select the formulas from the candidates that are within the short_range check for the respective neutral mass
        joined_df = pd.DataFrame()
        # Go through every adduct considered
        for adduct, mass_dev in adducts_to_consider.items():
            single_df = df2.loc[df2['Adduct'] == adduct]
            if len(single_df) > 0:
                neutral_mass = single_df.loc[:, 'Ad. Neutral Mass'].iloc[0]
                short_range = {'H/C': [short_range_eq_f(mass, short_range_eq['H/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['H/C'][1])],
                            'O/C': [short_range_eq_f(mass, short_range_eq['O/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['O/C'][1])],
                            'N/C': [short_range_eq_f(mass, short_range_eq['N/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['N/C'][1])],
                            'S/C': [short_range_eq_f(mass, short_range_eq['S/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['S/C'][1])],
                            'P/C': [short_range_eq_f(mass, short_range_eq['P/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['P/C'][1])],
                            'F/C': [short_range_eq_f(mass, short_range_eq['F/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['F/C'][1])],
                            'Cl/C': [short_range_eq_f(mass, short_range_eq['Cl/C'][0]),
                                    short_range_eq_f(mass, short_range_eq['Cl/C'][1])]}

                df_range = df2[(df2['H']/df2['C']) <= short_range['H/C'][1]]
                df_range = df_range[(df_range['H']/df_range['C']) >= short_range['H/C'][0]]
                df_range = df_range[(df_range['O']/df_range['C']) >= short_range['O/C'][0]]
                df_range = df_range[(df_range['O']/df_range['C']) <= short_range['O/C'][1]]
                df_range = df_range[(df_range['N']/df_range['C']) >= short_range['N/C'][0]]
                df_range = df_range[(df_range['N']/df_range['C']) <= short_range['N/C'][1]]
                df_range = df_range[(df_range['S']/df_range['C']) >= short_range['S/C'][0]]
                df_range = df_range[(df_range['S']/df_range['C']) <= short_range['S/C'][1]]
                df_range = df_range[(df_range['P']/df_range['C']) >= short_range['P/C'][0]]
                df_range = df_range[(df_range['P']/df_range['C']) <= short_range['P/C'][1]]
                df_range = df_range[(df_range['F']/df_range['C']) >= short_range['F/C'][0]]
                df_range = df_range[(df_range['F']/df_range['C']) <= short_range['F/C'][1]]
                df_range = df_range[(df_range['Cl']/df_range['C']) >= short_range['Cl/C'][0]]
                df_range = df_range[(df_range['Cl']/df_range['C']) <= short_range['Cl/C'][1]]

                # Add the candidates that are within the short range to the joined_df
                joined_df = pd.concat((joined_df, df_range))
        df_range = joined_df
        if len(df_range) != 0:
            if len(df_range) != len(df2):
                contribution['Check 1: Short Range'] = True
            df2 = df_range
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      df2.iloc[0].loc['Adduct'],
                      dict_iso,
                      'Check 1: Short Range',
                      contribution)#,
        #print(df2_range)
            
    if len(df2) > 1:   
        # Check nº 2: if there are formulas that can exist with elements in their most common valency
        # Con: Formulas with groups NO2 where N has a valency of 4 will be excluded
        df_valency = df2[df2['Valency'] == True]
        if len(df_valency) != 0:
            if len(df_valency) != len(df2):
                contribution['Check 2: Normal Valency Check'] = True
            df2 = df_valency
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      df2.iloc[0].loc['Adduct'],
                      dict_iso,
                      'Check 2: Normal Valency Check',
                      contribution)#,

        # Check nº 3 skipped

        # Check nº 3 part 2: if there are formulas with low amounts of different heteroatoms: prioritize C,H,O,N,S,P.
        df_CHONSP = df2[df2['F'] + df2['Cl'] == 0]
        if len(df_CHONSP) != 0:
            if len(df_CHONSP) != len(df2):
                contribution['Check 3-1: No F and Cl'] = True
            df2 = df_CHONSP
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      df2.iloc[0].loc['Adduct'],
                      dict_iso,
                      'Check 3-1: No F and Cl',
                      contribution)#,
        else:
            df_CHONSP = df2[df2['F'] + df2['Cl'] == (df2['F'] + df2['Cl']).min()]
            if len(df_CHONSP) != 0:
                if len(df_CHONSP) != len(df2):
                    contribution['Check 3-2: Lowest F + Cl'] = True
                df2 = df_CHONSP
                if len(df2) == 1:
                    if isotope_check == True:
                        dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                    return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                            df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                            False),
                        df2.index[0],
                        dict_iso,
                        df2.iloc[0].loc['Adduct'],
                        'Check 3-2: Lowest F + Cl',
                      contribution)#,

        if len(df2) > 1:
            # Check nº 4: Lowest Heteroatom count
            df3 = df2[df2[['N','S','P','F','Cl']].sum(axis = 1) ==
                      df2[['N','S','P','F','Cl']].sum(axis = 1).min()]
            if len(df3) != len(df2):
                contribution['Check 4: Lowest Heteroatom Count'] = True
            df2 = df3
            #df2 = df2[df2[['S','P']].sum(axis = 1) == 
            #          df2[['S','P']].sum(axis = 1).min()]
            if len(df2) == 1:
                if isotope_check == True:
                    dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

                return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],
                                        df2['S'].values[0],df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0],
                                        False),
                      df2.index[0],
                      df2.iloc[0].loc['Adduct'],
                      dict_iso,
                      'Check 4: Lowest Heteroatom Count',
                      contribution)#,

            # Final Check: Lowest error
            # Calculate and store the error (in ppm) of the mass of the filtered formulas
            joined_df = pd.DataFrame()
            # For every adduct type, calculate the mass deviation
            for adduct, mass_dev in adducts_to_consider.items():
                single_df = df2.loc[df2['Adduct'] == adduct]
                if len(single_df) > 0:
                    neutral_mass = single_df.loc[:, 'Ad. Neutral Mass'].iloc[0]
                    temp_df = pd.DataFrame(abs(((neutral_mass - single_df.index)/single_df.index)*1000000).values,
                                           index = single_df.index, columns = ['error'])
                    joined_df = pd.concat((joined_df, temp_df))
            mini = joined_df.idxmin(axis = 'index')
            contribution['Check 5: Lowest Mass error'] = True
            # Choose the formula with the lowest error (closest to original mass).
            if isotope_check == True:
                dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

            return(mass, formulator(df2.loc[mini, 'C'].values[0],df2.loc[mini,'H'].values[0],df2.loc[mini,'O'].values[0],
                                    df2.loc[mini,'N'].values[0],df2.loc[mini,'S'].values[0],df2.loc[mini,'P'].values[0],
                                    df2.loc[mini,'F'].values[0], df2.loc[mini,'Cl'].values[0],
                                    False),
                   mini.iloc[0],
                   df2.iloc[0].loc['Adduct'],
                   dict_iso,
                  'Check 5: Lowest Mass error',
                    contribution)

    # Only one formula is within the mass interval - it will be assigned
    elif len(df2) == 1:
        contribution['Check 0: Only 1 Formula Candidate'] = True
        if isotope_check == True:
            dict_iso = isotope_decider(dict_iso, isotopic_peaks, df2, mass, intensity, int_col, mass_column)

        return(mass, formulator(df2['C'].values[0],df2['H'].values[0],df2['O'].values[0],df2['N'].values[0],df2['S'].values[0],
                                df2['P'].values[0],df2['F'].values[0],df2['Cl'].values[0], False),
              df2.index[0],
              df2.iloc[0].loc['Adduct'],
              dict_iso,
              'Check 0: Only 1 Formula Candidate',
               contribution)