#!/usr/bin/env python

import numpy as np
import itertools
import sys

from cell2mol.tmcharge_common import atom, molecule, ligand, metal, group, getradii, getconec
from cell2mol.xyz2mol import int_atom, xyz2mol

from cell2mol.elementdata import ElementData

elemdatabase = ElementData()

#############################
### Loads Rdkit & xyz2mol ###
#############################

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdchem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import (
    MolDrawing,
    DrawingOptions,
)  # Only needed if modifying defaults

DrawingOptions.bondLineWidth = 2.2
from rdkit.Chem.rdmolops import SanitizeFlags
from rdkit.Chem import PeriodicTable
from pathlib import Path

# IPythonConsole.ipython_useSVG = False

from rdkit import rdBase

if "ipykernel" in sys.modules:
    try:
        from rdkit.Chem.Draw import IPythonConsole
    except ModuleNotFoundError:
        pass
#print("RDKIT Version:", rdBase.rdkitVersion)
rdBase.DisableLog("rdApp.*")


#######################################################
def classify_mols(moleclist, debug=0):

    # This subroutine reads all molecules and
    # identifies those that are identical based on a sort of ID number, which is
    # the variable "elemcountvec". This is a vector of occurrences for all elements in the elementdatabase

    molec_indices = []  # molecule index
    ligand_indices = []  # ligand index
    metal_ion_indices = []
    unique_indices = []  # Unique Specie (molecule or ligand) Index in Typelist
    unique_species = []  # List of Unique Species
    typelist_mols = []  # List of ID codes for Molecules
    typelist_ligs = []  # List of ID codes for Ligands
    typelist_mets = []

    # The reason for splitting typelist in two is that
    # some species might appear as ligands and as counterions, (eg. Cl-)
    # in the same unit cell. It is cleaner if these cases are treated separately

    # Note that both "typelists", and "unique_species" will be a list of tuples.
    specs_found = -1
    for idx, mol in enumerate(moleclist):
        if mol.type != "Complex":
            found = False
            for ldx, typ in enumerate(typelist_mols):
                if (mol.elemcountvec == typ[0]).all() and not found:
                    found = True
                    kdx = typelist_mols[ldx][1]
                    if debug >= 1:
                        print("Molecule", idx, "is the same than", ldx, "in typelist")
            if not found:
                specs_found += 1
                kdx = specs_found
                if debug >= 1:
                    print(
                        "New molecule found with:",
                        mol.labels,
                        "and added in position",
                        kdx,
                    )
                typelist_mols.append(list([mol.elemcountvec, kdx]))

                unique_species.append(list([mol.type, mol]))

            jdx = "-"
            molec_indices.append(idx)
            ligand_indices.append(jdx)
            unique_indices.append(kdx)

        elif mol.type == "Complex":
            for jdx, lig in enumerate(mol.ligandlist):
                found = False
                for ldx, typ in enumerate(typelist_ligs):
                    if (lig.elemcountvec == typ[0]).all() and not found:
                        found = True
                        kdx = typelist_ligs[ldx][1]
                        if debug >= 1:
                            print("Ligand", jdx, "is the same than", ldx, "in typelist")
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    if debug >= 1:
                        print(
                            "New ligand found with:",
                            lig.labels,
                            "and added in position",
                            kdx,
                        )
                    typelist_ligs.append(list([lig.elemcountvec, kdx]))
                    unique_species.append(list([lig.type, lig, mol]))

                molec_indices.append(idx)
                ligand_indices.append(jdx)
                unique_indices.append(kdx)

            for jdx, met in enumerate(mol.metalist):
                found = False
                for ldx, typ in enumerate(typelist_mets):
                    if (met.coord_sphere_ID == typ[0]).all() and not found:
                        found = True
                        kdx = typelist_mets[ldx][1]
                        if debug >= 1:
                            print("Metal", jdx, "is the same than", ldx, "in typelist")
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    if debug >= 1:
                        print(
                            "New Metal Center found with:",
                            met.label,
                            "and added in position",
                            kdx,
                        )
                    typelist_mets.append(list([met.coord_sphere_ID, kdx]))
                    unique_species.append(list([met.type, met, mol]))

                molec_indices.append(idx)
                ligand_indices.append(jdx)
                unique_indices.append(kdx)

    if debug >= 1:
        print("CLASSIFY: molec_indices", molec_indices)
    if debug >= 1:
        print("CLASSIFY: ligand_indices", ligand_indices)
    if debug >= 1:
        print("CLASSIFY: unique_indices", unique_indices)

    nspecs = len(unique_species)
    for idx in range(0,nspecs):
        count = unique_indices.count(idx)
        print(f"CLASSIFY: specie {idx} appears {count}")

    return molec_indices, ligand_indices, unique_indices, unique_species


#######################################################
def getcharge(labels, pos, initcharge, conmat, cov_factor=1.3, debug=0):
    ## Generates the connectivity of a molecule given a charge.
    # The molecule is described by the labels, and the atomic cartesian coordinates "pos"
    # The adjacency matrix is also provided (conmat)
    #:return iscorrect: boolean variable with a notion of whether the function delivered a good=True or bad=False connectivity
    #:return total_charge: total charge associated with the connectivity
    #:return atom_charge: atomic charge for each atom of the molecule
    #:return mols: rdkit molecule object
    #:return smiles: smiles representation of the molecule
    
    pt = Chem.GetPeriodicTable()    #needed to retrieve the default valences in the 2nd and 3rd checks
    natoms = len(labels)
    atnums = [int_atom(label) for label in labels]  # from xyz2mol

    ##########################
    # xyz2mol is called here #
    ##########################
    # use_graph is called for a faster generation
    # allow_charged_fragments is necessary for non-neutral molecules
    # embed_chiral shouldn't ideally be necessary, but it runs a sanity check that improves the proposed connectivity
    # use_huckel false means that the xyz2mol adjacency will be generated based on atom distances and vdw radii.
    # Ideally, the adjacency matrix could be provided
    mols = xyz2mol(
        atnums,
        pos,
        conmat,
        cov_factor,
        charge=initcharge,
        use_graph=True,
        allow_charged_fragments=True,
        embed_chiral=True,
        use_huckel=False,
    )
    if len(mols) > 1:
        print(
            "WARNING: More than 1 mol received from xyz2mol for initcharge:", initcharge
        )

    # smiles is generated with rdkit
    smiles = Chem.MolToSmiles(mols[0])

    # Here, the atom charge is retrieved, and the connectivity of each atom goes through 3 checks.
    # The variable iscorrect will track whether the overall generated structure is meaningful
    iscorrect = True
    atom_charge = []
    total_charge = 0
    for i in range(natoms):
        a = mols[0].GetAtomWithIdx(i)  # Returns a particular Atom
        atom_charge.append(a.GetFormalCharge())

        valence = a.GetTotalValence()  # Valence of the atom in the mol object
        bonds = 0
        countaromatic = 0
        for (
            b
        ) in (
            a.GetBonds()
        ):  # Returns a read-only sequence containing all of the molecule’s Bonds
            bonds += b.GetBondTypeAsDouble()
            # total number of bonds (weighted by bond order) of the atom in the mol object
            # Returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
            if b.GetBondTypeAsDouble() == 1.5:
                countaromatic += 1
        if countaromatic % 2 != 0:
            bonds -= 0.5

        total_charge += a.GetFormalCharge()
        lonepairs = (
            elemdatabase.valenceelectrons[a.GetSymbol()] - a.GetFormalCharge() - valence
        ) / 2
        totalvalenceelectrons = int(bonds) + int(lonepairs) * 2 + a.GetFormalCharge()

        # Checks the quality of the resulting smiles
        # First check, the number of lonepairs is computed and should make sense
        if (
            lonepairs != 0
            and lonepairs != 1
            and lonepairs != 2
            and lonepairs != 3
            and lonepairs != 4
        ):
            if debug >= 1:
                print("GETCHARGE: 2nd Check-lonepairs=", i, a.GetSymbol(), lonepairs)
            iscorrect = False

        # RDKIT has some troubles assigning the valence for atoms with aromatic bonds.
        # So the 2nd and 3rd Check applies only for countaromatic==0
        if countaromatic == 0:
            # Second check, the number of bonds should coincide with the valence.
            # I know it should be the same, but in bad SMILES they often do not coincide
            if bonds != valence:
                if debug >= 1:
                    print(
                        "GETCHARGE: 1st Check-bonds/valence:",
                        i,
                        a.GetSymbol(),
                        bonds,
                        valence,
                    )
                iscorrect = False
                if debug >= 1:
                    for b in a.GetBonds():
                        print(
                            b.GetBondTypeAsDouble(),
                            b.GetBeginAtomIdx(),
                            b.GetEndAtomIdx(),
                        )

            # Third check, using the totalvalenceelectrons
            if totalvalenceelectrons != elemdatabase.valenceelectrons[a.GetSymbol()]:
                if debug >= 1:
                    print(
                        "GETCHARGE: 3rd Check: Valence gives false for atom",
                        i,
                        a.GetSymbol(),
                        "with:",
                        totalvalenceelectrons,
                        elemdatabase.valenceelectrons[a.GetSymbol()],
                    )
                iscorrect = False

        if debug >= 1 and i == 0:
            print(
                "initcharge, atom idx, label, charge, pt.GetDefaultValence(a.GetAtomicNum()), valence, num bonds, num lonepairs, iscorrect"
            )
        if debug >= 1:
            print(
                initcharge,
                i,
                a.GetSymbol(),
                a.GetFormalCharge(),
                pt.GetDefaultValence(a.GetAtomicNum()),
                valence,
                int(bonds),
                int(lonepairs),
                iscorrect,
            )

    # if (debug == 1): print("")
    return iscorrect, total_charge, atom_charge, mols[0], smiles


#######################################################
def readchargelist(chargelist):
    abs_atlist = []
    abs_atcharge = []
    total = []
    zwitt = []
    for a in chargelist:
        tmp = []
        for b in a:
            tmp.append(abs(b))
        abs_atlist.append(tmp)
        abs_atcharge.append(np.sum(tmp))
        total.append(np.sum(a))

        if any(b > 0 for b in a) and any(b < 0 for b in a):
            zwitt.append(True)
        else:
            zwitt.append(False)

    abstotal = []
    for a in total:
        abstotal.append(abs(a))

    return total, abstotal, abs_atcharge, zwitt


#######################################################
def select_charge_distr(
    statuslist, uncorrected_chargelist, corrected_chargelist, chargestried, debug=0
):
    # This function selects the best charge_distribuion among the ones generated by the function "getcharge"
    # It does so based, in general, on the number of charges in each connectivity.
    #:return goodlist: list of acceptable charge distributions.
    # goodlist contains the indices of those distributions as the enter this function

    nlists = len(uncorrected_chargelist)

    uncorr_total, uncorr_abs_total, uncorr_abs_atcharge, uncorr_zwitt = readchargelist(
        uncorrected_chargelist
    )
    corr_total, corr_abs_total, corr_abs_atcharge, corr_zwitt = readchargelist(
        corrected_chargelist
    )

    if debug >= 1:
        print("    NEW SELECT FUNCTION: uncorr_total:", uncorr_total)
    if debug >= 1:
        print("    NEW SELECT FUNCTION: uncorr_abs_total:", uncorr_abs_total)
    if debug >= 1:
        print("    NEW SELECT FUNCTION: uncorr_abs_atcharge:", uncorr_abs_atcharge)
    if debug >= 1:
        print("    NEW SELECT FUNCTION: uncorr_zwitt:", uncorr_zwitt)

    coincide = []
    for idx, a in enumerate(uncorr_total):
        if a == chargestried[idx]:
            coincide.append(True)
        if a != chargestried[idx]:
            coincide.append(False)

    minoftot = np.min(uncorr_abs_total)
    minofabs = np.min(uncorr_abs_atcharge)
    listofmintot = [i for i, x in enumerate(uncorr_abs_total) if x == minoftot]
    listofminabs = [i for i, x in enumerate(uncorr_abs_atcharge) if x == minofabs]

    if debug >= 1:
        print("    NEW SELECT FUNCTION: listofmintot:", listofmintot)
    if debug >= 1:
        print("    NEW SELECT FUNCTION: listofminabs:", listofminabs)
    if debug >= 1:
        print("    NEW SELECT FUNCTION: coincide:", coincide)

    # Searches for entries that have the smallest total charge(appear in listofmintot),
    #     and smallest number of charges(appear in listofminabs)
    tmplist = []
    for idx in range(0, nlists):
        if (idx in listofminabs) and (idx in listofmintot) and coincide[idx]:
            tmplist.append(idx)

    if debug >= 1:
        print("    NEW SELECT FUNCTION: tmplist:", tmplist)

    # CASE 1, IF only one distribution meets the requirement. Then it is chosen
    if len(tmplist) == 1:
        goodlist = tmplist.copy()
        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Case 1, only one entry in tmplist, so goodlist is:",
                goodlist,
            )

    # CASE 2, IF listofminabs and listofmintot do not have any value in common. Then we select from minima, coincide, and zwitt
    elif len(tmplist) == 0:
        goodlist = []
        for idx in range(0, nlists):
            if (
                (idx in listofminabs)
                or (idx in listofmintot)
                and coincide[idx]
                and not uncorr_zwitt[idx]
            ):
                goodlist.append(idx)
        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Case 2, no entry in initial tmplist. We select from minima, coincide and zwitt:",
                goodlist,
            )

    # CASE 3, IF more than one in list is correct. Then, I choose the one that delivered the expected charge and that delivers the minimum charge after H removal
    elif len(tmplist) > 1:

        if debug >= 1:
            print("    NEW SELECT FUNCTION: Initial corr_total:", corr_total)
        if debug >= 1:
            print("    NEW SELECT FUNCTION: Initial corr_abs_total:", corr_abs_total)
        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Initial corr_abs_atcharge:", corr_abs_atcharge
            )

        selected_corr_abs_total = []
        selected_corr_abs_atcharge = []
        indices_transfer = []
        for idx in range(0, nlists):
            if idx in tmplist and coincide[idx]:
                selected_corr_abs_total.append(corr_abs_total[idx])
                selected_corr_abs_atcharge.append(corr_abs_atcharge[idx])
                indices_transfer.append(idx)

        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Initial selected_corr_total:",
                selected_corr_abs_total,
            )
        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Initial selected_corr_abs_total:",
                selected_corr_abs_atcharge,
            )

        ncorr_list = len(selected_corr_abs_total)

        ####### Option 1, go for minimum charge of the corrected one
        minoftot = np.min(selected_corr_abs_total)
        minofabs = np.min(selected_corr_abs_atcharge)
        listofmintot = [
            i for i, x in enumerate(selected_corr_abs_total) if x == minoftot
        ]
        listofminabs = [
            i for i, x in enumerate(selected_corr_abs_atcharge) if x == minofabs
        ]

        if debug >= 1:
            print("    NEW SELECT FUNCTION: sel_listofmaxtot:", listofmintot)
        if debug >= 1:
            print("    NEW SELECT FUNCTION: sel_listofmaxabs:", listofminabs)

        goodlist = []
        for idx in range(0, ncorr_list):
            if (idx in listofmintot) and (idx in listofminabs):
                goodlist.append(indices_transfer[idx])

        if debug >= 1:
            print(
                "    NEW SELECT FUNCTION: Case 3, multiple entries in tmplist, so goodlist is:",
                goodlist,
            )

    ###### CASE 4. IF, at this stage, a clear option is not found. Then, resort to coincide. Even if the charge works, the connectivity is probably wrong
    if len(goodlist) == 0:
        if debug >= 1:
            print(
                "    SELECT FUNCTION: Case 4, empty goodlist so going for coincide as our last resort"
            )
        for idx, g in enumerate(coincide):
            if g:
                goodlist.append(idx)

    return goodlist


#######################################################
def get_poscharges_unique_species(unique_species):

    for idx, spec in enumerate(unique_species):
        # Obtains charge for Organic Molecules
        if spec[0] == "Other":
            mol = spec[1]
            print("    ---------------")
            print("    #### NON-Complex ####")
            print("    ---------------")
            print("          ", mol.natoms, mol.labels)
            (
                mol.poscharge,
                mol.posatcharge,
                mol.posobjlist,
                mol.possmiles,
                Warning,
            ) = get_poscharges_organic(mol)
            if Warning:
                print("Empty possible charges received for molecule", mol.labels)

        # Obtains charge for Ligands in Complexes
        elif spec[0] == "Ligand":
            lig = spec[1]
            mol = spec[2]
            print("")
            print("    ---------------")
            print("    #### Ligand ####")
            print("    ---------------")
            print("          ", lig.natoms, lig.labels, lig.totmconnec)
            # isinlibrary = getpresets(lig)
            (
                lig.poscharge,
                lig.posatcharge,
                lig.posobjlist,
                lig.possmiles,
                Warning,
            ) = get_poscharges_tmcomplex(lig, mol)
            if Warning:
                print("Empty possible charges received for ligand", lig.labels)

        # Adds possible Charges for Metals
        elif spec[0] == "Metal":
            met = spec[1]
            mol = spec[2]
            print("")
            print("    ---------------")
            print("    #### Metal ####")
            print("    ---------------")
            print("          ", met.label, met.coord_sphere)
            met.poscharge = get_metal_poscharge(met.label)

            list_of_zero_OS = ["Fe", "Ni", "Ru"]
            if met.label in list_of_zero_OS:
                # In some cases, it adds 0 as possible metal charge
                # -if it has CO ligands
                if any(
                    (lig.natoms == 2 and "C" in lig.labels and "O" in lig.labels)
                    for lig in mol.ligandlist
                ):
                    if int(0) not in met.poscharge:
                        met.poscharge.append(int(0))
                # -if it has any ligand with hapticity
                if any((lig.hapticity) for lig in mol.ligandlist):
                    if int(0) not in met.poscharge:
                        met.poscharge.append(int(0))

            print("Possible charges received for metal:", met.poscharge)

    return unique_species, Warning


#######################################################
def get_poscharges_organic(mol):

    debug = 0
    Warning = False
    maxfragcharge = int(
        3
    )  # maximum expected charge for non-complex (i.e. organic) molecules

    addedH = 0
    statuslist = []
    fchlist = []
    chlist = []
    smileslist = []
    molobjlist = []
    # addedlist = np.zeros((len(mol.natoms))).astype(int)

    poscharge = mol.poscharge
    posatcharge = mol.posatcharge
    posobjlist = mol.posobjlist
    possmiles = mol.possmiles

    # The mol objects of molecules are generated based on several possible charges (variable ich)
    chargestried = []
    for magn in range(0, int(maxfragcharge + 1)):
        if magn == 0:
            signlist = [1]
        if magn != 0:
            signlist = [-1, 1]
        for sign in signlist:
            ich = int(magn * sign)  # chargestried
            # if magn == 0: print("    MAIN: charge sending", mol.labels, mol.coord, ich, mol.factor)
            status, fch, ch, molob, smiles = getcharge(
                mol.labels, mol.coord, ich, mol.conmat, mol.factor
            )
            # iscorrect, total_charge, atom_charge, mols[0], smiles
            # These variables contain all results of assigning the charge above
            if status:
                chargestried.append(ich)
                statuslist.append(status)
                fchlist.append(fch)  # total_charge
                chlist.append(ch)  # atom_charge
                smileslist.append(smiles)
                molobjlist.append(molob)
                if debug >= 1:
                    print("    MAIN: charge", ich, "- smiles", smiles)
            elif not status:
                if debug >= 1:
                    print("    MAIN: Wrong status for charge", ich, "- smiles", smiles)
                if debug >= 1:
                    print("    MAIN:", status, fch, ch)

    # all fchlist and chlist are collected, so are sent to choose the best distribution of charges among atoms
    if len(statuslist) > 0:
        uncorr_chlist = chlist.copy()
        gooddistr = select_charge_distr(
            statuslist, uncorr_chlist, chlist, chargestried, debug
        )
    else:
        gooddistr = []
        print("    SELECT FUNCTION: Empty lists sent to select_charge_distr. Stopping")
        Warning = True

    # Treats (adds/changes) the molecular properties depending on the result of the "select_charge_distr" function
    if len(gooddistr) >= 1:  # Means that only one option has been found
        for idx, g in enumerate(gooddistr):
            if fchlist[g] not in mol.poscharge:
                print("    MAIN. gooddistr:", g, "with Charge:", fchlist[g])
                poscharge.append(fchlist[g])
                posatcharge.append(chlist[g])
                posobjlist.append(molobjlist[g])
                possmiles.append(smileslist[g])
                print("    MAIN. poscharge added:", fchlist[g])
    elif len(gooddistr) == 0:  # Means that no good option has been found
        Warning = True

    return poscharge, posatcharge, posobjlist, possmiles, Warning


#######################################################
def get_poscharges_tmcomplex(lig, mol, debug=1):
    # This function drives the detrmination of the charge for a "ligand=lig" of a given "molecule=mol"
    # The whole process is done by the functions:
    # 1) define_sites, which determines which atoms must have added elements (see above)
    # 2) once any atom is added, the function "getcharge" generates one connectivity for a set of charges
    # 3) select_charge_distr chooses the best connectivity among the generated ones.

    # Basically, this function connects these other three functions,
    # while managing some key information for those

    Warning = False

    #### Educated Guess on the Maximum Charge one can expect from the ligand
    if not lig.hapticity:
        maxfragcharge = lig.totmconnec + 2
    if lig.hapticity:
        maxfragcharge = 2

    # Cases of same atom being connected to more than one metal
    if any(a.mconnec >= 2 for a in lig.atoms):
        pass
    else:
        if maxfragcharge > lig.natoms:
            maxfragcharge = lig.natoms
    if maxfragcharge > int(5):
        maxfragcharge = 5

    #### Adds Hydrogens depending on whether there is Hapticity or not
    tmplab, tmpcoord, addedH, addedlist, metal_electrons = define_sites(
        lig, mol.metalist, mol, debug
    )

    # Initiates Variables Needed to Select Charge
    statuslist = []
    fchlist = []
    uncorr_fchlist = []  # Uncorrected lists = without the H correction to attoms
    uncorr_chlist = []
    chlist = []
    smileslist = []
    molobjlist = []

    poscharge = lig.poscharge
    posatcharge = lig.posatcharge
    posobjlist = lig.posobjlist
    possmiles = lig.possmiles

    tmpradii = getradii(tmplab)
    dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(tmplab, tmpcoord, lig.factor, tmpradii)

    # Launches getcharge for each initial charge
    chargestried = []
    for magn in range(0, int(maxfragcharge + 1)):
        if magn == 0:
            signlist = [1]
        if magn != 0:
            signlist = [-1, 1]
        for sign in signlist:
            ich = int(magn * sign)
            try:
                status, uncorr_fch, uncorr_ch, molob, smiles = getcharge(
                    tmplab, tmpcoord, ich, tmpconmat, lig.factor
                )
                chargestried.append(ich)

                # Corrects the Charge of atoms with addedH
                ch = []
                count = 0
                for i, a in enumerate(
                    lig.atoms
                ):  # Iterates over the original number of ligand atoms, thus without the added H
                    if addedlist[i] != 0:
                        count += 1
                        ch.append(
                            uncorr_ch[i]
                            - addedlist[i]
                            + metal_electrons[i]
                            - uncorr_ch[lig.natoms - 1 + count]
                        )
                        # last term corrects for cases in which a charge has been assigned to the added atom
                        # ch.append(uncorr_ch[i]-addedlist[i]+metal_electrons[i])
                    else:
                        ch.append(uncorr_ch[i])
                fch = np.sum(ch)

                statuslist.append(status)
                fchlist.append(fch)
                chlist.append(ch)
                uncorr_fchlist.append(uncorr_fch)
                uncorr_chlist.append(uncorr_ch)
                smileslist.append(smiles)
                molobjlist.append(molob)

                if debug == 1:
                    print("    MAIN: charge", ich, "- smiles", smiles)
            except Exception as exc:
                print("exception in get_poscharges_tmcomplex:", exc)
                pass

    #### Evaluates possible charges except if the ligand is a nitrosyl
    if lig.natoms == 2 and "N" in lig.labels and "O" in lig.labels:
        NO_type = get_nitrosyl_geom(lig)
        if NO_type == "Linear":
            gooddistr = [2]
            print("    MAIN. gooddistr:", gooddistr)
        elif NO_type == "Bent":
            gooddistr = [0]
            print("    MAIN. gooddistr:", gooddistr)
    else:
        if len(statuslist) > 0:
            gooddistr = select_charge_distr(
                statuslist, uncorr_chlist, chlist, chargestried, debug
            )
            print("    MAIN. gooddistr:", gooddistr, "with Charge(s):")
        else:
            gooddistr = []
            print("    MAIN. Empty lists sent to select_charge_distr. Stopping")
            Warning = True

    # Treats (adds/changes) the molecular properties depending on the result of the "select_charge_distr" function
    if len(gooddistr) >= 1:  # Means that one or more options have been found
        for idx, g in enumerate(gooddistr):
            if fchlist[g] not in lig.poscharge:
                print(
                    "    MAIN. gooddistr:",
                    g,
                    "with Charge:",
                    fchlist[g],
                    "uncorrected:",
                    uncorr_fchlist[g],
                )
                poscharge.append(fchlist[g])
                posatcharge.append(chlist[g])
                posobjlist.append(molobjlist[g])
                possmiles.append(smileslist[g])
                print("    MAIN. poscharge added:", fchlist[g])
    elif len(gooddistr) == 0:  # Means that no good option has been found
        Warning = True

    return poscharge, posatcharge, posobjlist, possmiles, Warning


#######################################################
def get_nitrosyl_geom(ligand):
    # Function that determines whether the M-N-O angle of a Nitrosyl "ligand" is "Bent" or "Linear"
    # Each case is treated differently
    #:return NO_type: "Linear" or "Bent"

    from cell2mol.missingH import getangle

    debug = 0

    for idx, a in enumerate(ligand.atoms):
        if a.label == "N":
            central = a.coord.copy()
        if a.label == "O":
            extreme = a.coord.copy()

    dist = []
    for idx, met in enumerate(ligand.metalatoms):
        metal = np.array(met.coord)
        dist.append(np.linalg.norm(central - metal))
    tgt = np.argmin(dist)

    metal = ligand.metalatoms[tgt].coord.copy()

    if debug >= 1:
        print("NITRO coords:", central, extreme, metal)

    vector1 = np.subtract(np.array(central), np.array(extreme))
    vector2 = np.subtract(np.array(central), np.array(metal))

    if debug >= 1:
        print("NITRO Vectors:", vector1, vector2)

    angle = getangle(vector1, vector2)
    if debug >= 1:
        print("NITRO ANGLE:", angle, np.degrees(angle))

    if np.degrees(angle) > float(160):
        NO_type = "Linear"
    else:
        NO_type = "Bent"

    return str(NO_type)


#######################################################
def get_metal_poscharge(label):
    ## Retrieves plausible oxidation states for a given metal

    from collections import defaultdict

    # Data Obtained from:
    # Venkataraman, D.; Du, Y.; Wilson, S. R.; Hirsch, K. A.; Zhang, P.; Moore, J. S. A
    # Coordination Geometry Table of the D-Block Elements and Their Ions.
    # J. Chem. Educ. 1997, 74, 915.

    atnum = elemdatabase.elementnr[label]

    at_charge = defaultdict(list)
    # 1st-row transition metals.
    at_charge[21] = [3]  # Sc
    at_charge[22] = [2, 3, 4]  # Ti
    at_charge[23] = [1, 2, 3, 4]  # V
    at_charge[24] = [0, 2, 3]  # Cr
    at_charge[25] = [1, 2, 3]  # Mn
    at_charge[26] = [2, 3]  # Fe
    at_charge[27] = [1, 2, 3]  # Co
    at_charge[28] = [2, 3]  # Ni
    at_charge[29] = [1, 2]  # Cu
    at_charge[30] = [2]  # Zn
    # 2nd-row transition metals.
    at_charge[39] = [3]  # Y
    at_charge[40] = [2, 3, 4]  # Zr
    at_charge[41] = [5, 4, 3]  # Nb
    at_charge[42] = [0, 2, 3, 4, 5, 6]  # Mo
    at_charge[43] = [1, 2, 3, 4]  # Tc
    at_charge[44] = [2, 3]  # Ru
    at_charge[45] = [1, 2, 3]  # Rh
    at_charge[46] = [2]  # Pd
    at_charge[47] = [1]  # Ag
    at_charge[48] = [2]  # Cd
    # 3rd-row transition metals.
    at_charge[57] = []  # La
    at_charge[72] = [4]  # Hf
    at_charge[73] = [2, 3, 4]  # Ta
    at_charge[74] = [0, 2, 3, 4, 5, 6]  # W
    at_charge[75] = [1, 2, 3, 4, 5]  # Re
    at_charge[76] = [0, 2, 3, 4, 5]  # Os
    at_charge[77] = [1, 3]  # Ir
    at_charge[78] = [2, 4]  # Pt
    at_charge[79] = [1, 3]  # Au
    at_charge[80] = [2]  # Hg

    return at_charge[atnum]


#######################################################
def balance_charge(
    moleclist,
    molec_indices,
    ligand_indices,
    unique_indices,
    unique_species,
    typ_charge="Primary",
):

    # Function to Select the Best Charge Distribution for the unique species.
    # It accepts multiple charge options for each molecule/ligand/metal (poscharge, etc...).
    # NO: It should select the best one depending on whether the final metal charge makes sense or not.
    # In some cases, can accept metal oxidation state = 0, if no other makes sense

    debug = 0
    Error = False

    iterlist = []
    for idx, spec_tuple in enumerate(unique_species):
        spec = spec_tuple[1]
        toadd = []
        if len(spec.poscharge) == 1:
            toadd.append(spec.poscharge[0])
        elif len(spec.poscharge) > 1:
            for tch in spec.poscharge:
                toadd.append(tch)
        elif len(spec.poscharge) == 0:
            Error = True
            toadd.append("-")
        if typ_charge == "Secondary":
            toadd.append(int(0))
        iterlist.append(toadd)

    if debug >= 1:
        print("BALANCE: iterlist", iterlist)
    if debug >= 1:
        print("BALANCE: unique_indices", unique_indices)

    if not Error:
        tmpdistr = list(itertools.product(*iterlist))
        if debug >= 1:
            print("BALANCE: tmpdistr", tmpdistr)

        # Expands tmpdistr to include same species, generating alldistr:
        alldistr = []
        for distr in tmpdistr:
            tmp = []
            for u in unique_indices:
                tmp.append(distr[u])
            alldistr.append(tmp)
            if debug >= 1:
                print("BALANCE: alldistr added:", tmp)

            gooddistr = []
            for idx, d in enumerate(alldistr):
                final_charge = np.sum(d)
                if final_charge == 0:
                    gooddistr.append(d)
    elif Error:
        print("Error found in balance_charge: one species has no possible charges")
        gooddistr = []

    return gooddistr


#######################################################
def define_sites(ligand, metalist, molecule, debug=1):
    # This function is the heart of the program.
    # It decides whether a connected atom must be temporarily added an extra element
    # when running the getcharge function
    # It does so in order to generate a meaningful connectivity for the ligand,
    # without any additional difficulty due to the missing M-L bond(s)

    # The function works on all "groups" of a ligand,
    # which are groups of atoms that are all connected among themselves and to the metal
    # For instance, in a Cp ligand, all C atoms belong to the same group.
    # Also, for the ligand CN, C is the only group

    # Groups are identified in the part of the code that does the cell reconstruction, and have properties.
    # One of this properties is whether they form haptic bonds.
    # Haptic groups require a catalogue of rules, since they behave differently from regular groups (i.e. non-haptic)
    # These rules are defined below.

    # If a group is not haptic. Then rules are applied depending on the type of the connected atom.
    # Ideally, these rules can be defined depending on the adjacency of the atom.
    # In some cases, this is not the case, and so we resort to an actual connectivity,
    # in what I call a non-local approach.

    # The non-local approach consists of generating a tentative connectivity using xyz2mol, assuming a neutral charge.
    # This is not necessarily true, but hopefully the connectivity of the connected atom will be correct.

    newlab = ligand.labels.copy()
    newcoord = ligand.coord.copy()

    # Variables that control how many atoms have been added.
    tmp_added_atoms = 0

    # Boolean that decides whether a non-local approach is needed
    needs_nonlocal = False

    # Initialization of the variables
    addedlist = np.zeros((len(newlab))).astype(int)
    block = np.zeros((len(newlab))).astype(int)
    metal_electrons = np.zeros((len(newlab))).astype(int)  # It will remain as such
    elemlist = np.empty((len(newlab))).astype(str)

    # Program runs sequentially for each group of the ligand
    for g in ligand.grouplist:

        ########################
        # Cases with Hapticity #
        ########################
        if len(g.hapttype) > 0:  # then the group has hapticity
            Selected_Hapticity = False
            if debug >= 1:
                print(
                    "        DEFINE_SITES: addressing group with hapticity:", g.hapttype
                )

            if "h5-Cp" in g.hapttype and not Selected_Hapticity:
                if debug >= 1:
                    print("        DEFINE_SITES: It is an h5-Cp with atlist:", g.atlist)
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    # print(idx, a.label, a.mconnec)
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                            # print("added")
                        else:
                            block[idx]

            elif "h7-Cicloheptatrienyl" in g.hapttype and not Selected_Hapticity:
                if debug >= 1:
                    print("        DEFINE_SITES: It is an h7-Cicloheptatrienyl")
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx]

            elif "h5-AsCp" in g.hapttype and not Selected_Hapticity:
                if debug >= 1:
                    print("        DEFINE_SITES: It is an h5-AsCp")
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx]

            elif (
                "h3-Allyl" in g.hapttype or "h3-Cp" in g.hapttype
            ) and not Selected_Hapticity:
                # if "h3-Allyl" or "h3-Cp" in g.hapttype:
                if debug >= 1:
                    print("        DEFINE_SITES: It is either an h3-Allyl or an h3-Cp")
                Selected_Hapticity = True
                tobeadded = 1
                tmp_added_atoms = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        if tmp_added_atoms < tobeadded:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                            tmp_added_atoms += 1
                        else:
                            block[idx]

            elif (
                "h4-Benzene" in g.hapttype or "h4-Butadiene" in g.hapttype
            ) and not Selected_Hapticity:
                if debug >= 1:
                    print(
                        "        DEFINE_SITES: It is either an h4-Benzene or an h4-Butadiene"
                    )
                if debug >= 1:
                    print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx]

            elif (
                "h2-Benzene" in g.hapttype or "h2-Butadiene" in g.hapttype
            ) and not Selected_Hapticity:
                if debug >= 1:
                    print(
                        "        DEFINE_SITES: It is either an h2-Benzene or an h2-Butadiene"
                    )
                if debug >= 1:
                    print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx]

            elif "h4-Enone" in g.hapttype and not Selected_Hapticity:
                if debug >= 1:
                    print("        DEFINE_SITES: It is an h4-Enone")
                if debug >= 1:
                    print("        DEFINE_SITES: No action is required")
                Selected_Hapticity = True
                tobeadded = 0
                for idx, a in enumerate(ligand.atoms):
                    if idx in g.atlist and a.mconnec == 1:
                        block[idx]

            # If the group hapticity type is not recognized -or instructions are not defined-, nothing is done
            if not Selected_Hapticity:
                if debug >= 1:
                    print(
                        f"        DEFINE_SITES: {g.hapttype} not recognized or new rules are necessary"
                    )

        else:  # cases without hapticity
            ions = [
                "F",
                "Cl",
                "Br",
                "I",
                "As",
            ]  # Atoms for which an H atom is always added
            ###########################
            # Cases with No Hapticity #
            ###########################
            # An initial attempt to add elements based on the adjacency of the connected atom
            for idx in g.atlist:
                a = ligand.atoms[idx]
                if debug >= 1:
                    print(
                        f"        DEFINE_SITES: evaluating non-haptic group with index {idx} and label {a.label}"
                    )
                # Simple Ionic Case
                if a.label in ions:
                    elemlist[idx] = "H"
                    addedlist[idx] = 1
                # Oxigen
                elif a.label == "O" or a.label == "S" or a.label == "Se":
                    if a.connec == 1:
                        needs_nonlocal = True
                        if debug >= 1:
                            print(
                                f"        DEFINE_SITES: will be sent to nonlocal due to {a.label} atom"
                            )
                    else:
                        block[idx] = 1
                # I'm considering a different one with S and Se
#                 elif a.label == "S" or a.label == "Se":
#                     if a.connec == 1:
#                         elemlist[idx] = "H"
#                         addedlist[idx] = 1
                # Hydrides
                elif a.label == "H":
                    if a.connec == 0:
                        elemlist[idx] = "Cl"
                        addedlist[idx] = 1
                    else:
                        block[idx] = 1
                # Nitrogen
                elif a.label == "N":
                    # Nitrosyl
                    if ligand.natoms == 2 and "O" in ligand.labels:
                        NO_type = get_nitrosyl_geom(ligand)
                        if NO_type == "Linear":
                            if debug >= 0:
                                print("        DEFINE_SITES: Found Linear Nitrosyl")
                            elemlist[idx] = "O"
                            addedlist[idx] = 2
                            metal_electrons[idx] = 1
                        elif NO_type == "Bent":
                            if debug >= 0:
                                print("        DEFINE_SITES: Found Bent Nitrosyl")
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                    else:
                        # nitrogen with at least 3 adjacencies doesnt need H
                        if a.connec >= 3:
                            block[idx] = 1
                        else:
                            # Checks for adjacent Atoms
                            list_of_adj_atoms = []
                            for i in a.adjacency:
                                list_of_adj_atoms.append(ligand.labels[i])
                            numN = list_of_adj_atoms.count("N")
                            if numN == 2:  # triazole or tetrazole
                                elemlist[idx] = "H"
                                addedlist[idx] = 1
                            else:
                                needs_nonlocal = True
                # Phosphorous
                elif (a.connec == 3) and a.label == "P":
                    block[idx] = 1
                # Case of Carbon (Simple CX vs. Carbenes)
                elif a.label == "C":
                    if ligand.natoms == 2:
                        # CN
                        if "N" in ligand.labels:
                            elemlist[idx] = "H"
                            addedlist[idx] = 1
                        # CO
                        if "O" in ligand.labels:
                            block[idx] = 1
                    # Added in V15e for amides
                    elif (
                        any(ligand.labels[i] == "O" for i in a.adjacency)
                        and any(ligand.labels[i] == "N" for i in a.adjacency)
                        and a.connec == 2
                    ):
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        iscarbene, tmp_element, tmp_added, tmp_metal = check_carbenes(
                            a, ligand, molecule
                        )
                        if debug >= 1:
                            print(
                                "        DEFINE_SITES: Evaluating as carbene and",
                                iscarbene,
                            )
                        if iscarbene:
                            # Carbene identified
                            elemlist[idx] = tmp_element
                            addedlist[idx] = tmp_added
                            metal_electrons[idx] = tmp_metal
                        else:
                            if not needs_nonlocal:
                                needs_nonlocal = True
                                if debug >= 1:
                                    print(
                                        "        DEFINE_SITES: will be sent to nonlocal due to C atom"
                                    )
                # Silicon
                elif a.label == "Si":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        block[idx]
                # Boron
                elif a.label == "B":
                    if a.connec < 4:
                        elemlist[idx] = "H"
                        addedlist[idx] = 1
                    else:
                        block[idx]
                # None of the previous options
                else:
                    if not needs_nonlocal:
                        needs_nonlocal = True
                        if debug >= 1:
                            print(
                                "        DEFINE_SITES: will be sent to nonlocal due to connected atom with no rules"
                            )

        # If, at this stage, we have found that any atom must be added, this is done before entering the non_local part.
        # The block variable makes that more atoms cannot be added to these connected atoms
        added_atoms = 0
        for idx, a in enumerate(ligand.atoms):
            if addedlist[idx] != 0 and block[idx] == 0:
                isadded, newlab, newcoord = add_atom(newlab, newcoord, idx, ligand, metalist, elemlist[idx]                )
                block[idx] = 1  # No more elements will be added to those atoms
                if isadded: 
                    added_atoms += addedlist[idx]
                    if debug >= 1: print(f"        DEFINE_SITES: Added {elemlist[idx]} to atom {idx} with: a.mconnec={a.mconnec} and label={a.label}")
                else: 
                   addedlist[idx] = 0

    ############################
    ###### NON-LOCAL PART ######
    ############################
    # In some cases, the decision to add an element to a connected atom cannot be taken based on its adjacencies.
    # Then, a preliminary bond-order connectivity is generated with rdkit using charge=0.
    # this connectivity can contain errors, but is tipically enough to determine the bonds of the connected atom with the rest of the ligand
    if needs_nonlocal:
        if debug >= 1:
            print(f"        DEFINE_SITES: Enters non-local")
        if debug >= 1:
            print(f"        DEFINE_SITES: block:", block)

        avoid = ["Si", "P"]
        metal_electrons = np.zeros((len(newlab))).astype(int)  ## Electrons Contributed to the Metal
        elemlist = np.empty((len(newlab))).astype(str)

        natoms = len(newlab)
        atnums = [int_atom(label) for label in newlab]  # from xyz2mol
        pos = newcoord.copy()

        # Generate the new adjacency matrix to be sent to xyz2mol
        tmpradii = getradii(newlab)
        dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(newlab, pos, ligand.factor, tmpradii)

        # Generation of the tentative neutral connectivity for the ligand. Here we use allow_charged_fragments = False
        try:
            tmpmol = xyz2mol(
                atnums,
                pos,
                tmpconmat,
                ligand.factor,
                charge=0,
                use_graph=True,
                allow_charged_fragments=False,
                embed_chiral=False,
                use_huckel=False,
            )
        except Exception as m:
            tmpmol = []
            for idx, a in enumerate(newlab):
                print("%s  %.6f  %.6f  %.6f" % (a, pos[idx][0], pos[idx][1], pos[idx][2])) 

        if len(tmpmol) > 0:
            for mol in tmpmol:
                smi = Chem.MolToSmiles(mol)
                print("        DEFINE_SITES: TMP smiles:", smi)
            for idx, a in enumerate(ligand.atoms):
                addH = False
                if a.mconnec == 1 and a.label not in avoid:
                    rdkitatom = tmpmol[0].GetAtomWithIdx(idx)
                    conjugated = False
                    total_bond_order = 0
                    bondlist = []
                    for b in rdkitatom.GetBonds():
                        bond = b.GetBondTypeAsDouble()
                        bondlist.append(bond)
                        total_bond_order += bond
                    if (
                        a.label == "O" or a.label == "S" or a.label == "Se"
                    ) and total_bond_order < 2:
                        addH = True
                    elif a.label == "N" and total_bond_order < 3:
                        addH = True
                    elif a.label == "C" and total_bond_order < 4:
                        addH = True
                    if debug >= 1:
                        print(
                            f"        DEFINE_SITES: Non-Local reports: {total_bond_order} for atom {idx} with: a.mconnec={a.mconnec} and label={a.label}"
                        )
                if addH and block[idx] == 0:
                    elemlist[idx] = "H"
                    added_atoms = added_atoms + 1
                    addedlist[idx] = 1

        # Adds the elements to the non_local atoms
        for idx, a in enumerate(ligand.atoms):
            if addedlist[idx] != 0 and block[idx] == 0:
                isadded, newlab, newcoord = add_atom(newlab, newcoord, idx, ligand, metalist, elemlist[idx])
                block[idx] = 1
                if isadded: 
                    added_atoms += addedlist[idx]
                    if debug >= 1: print(f"        DEFINE_SITES: Added {elemlist[idx]} to atom {idx} with: a.mconnec={a.mconnec} and label={a.label}")
                else: 
                   addedlist[idx] = 0
        if debug >= 1: print(f"        DEFINE_SITES: {added_atoms} H atoms added to ligand with addedlist={addedlist}")

    return newlab, newcoord, added_atoms, addedlist, metal_electrons  # tmplab, tmpcoord


#######################################################
def add_atom(labels, coords, site, ligand, metalist, element="H", debug=1):
    # This function adds one atom of a given "element" to a given "site=atom index" of a "ligand".
    # It does so at the position of the closest "metal" atom to the "site"
    #:return newlab: labels of the original ligand, plus the label of the new element
    #:return newcoord: same as above but for coordinates

    # Original labels and coordinates are copied
    posadded = len(labels)
    newlab = labels.copy()
    newcoord = coords.copy()
    newlab.append(str(element))  # One H atom will be added
    isadded = False

    if debug >= 2:
        print("        ADD_ATOM: Metalist length", len(metalist))
    # It is adding the element (H, O, or whatever) at the vector formed by the closest TM atom and the "site"
    for idx, a in enumerate(ligand.atoms):
        if idx == site:
            apos = np.array(a.coord)
            dist = []
            if debug >= 2:
                print("        ADD_ATOM: Atom coords:", apos)
            for tm in metalist:
                bpos = np.array(tm.coord)
                dist.append(np.linalg.norm(apos - bpos))
                if debug >= 2:
                    print("        ADD_ATOM: Metal coords:", bpos)

            # finds the closest Metal Atom (tgt), and adds element at the distance determined by the two elements vdw radii
            tgt = np.argmin(dist)
            idealdist = a.radii + elemdatabase.CovalentRadius2[element]
            addedHcoords = apos + (metalist[tgt].coord - apos) * (idealdist / dist[tgt]
            )  # the factor idealdist/dist[tgt] controls the distance
            newcoord.append([addedHcoords[0], addedHcoords[1], addedHcoords[2]])  # adds H at the position of the closest Metal Atom

            # Evaluates the new adjacency matrix. 
            tmpradii = getradii(newlab)
            dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(newlab, newcoord, ligand.factor, tmpradii)
            # If no undesired adjacencies have been created, the coordinates are kept
            if tmpconnec[posadded] == 1:
               isadded = True
               if debug >= 1: print(f"        ADD_ATOM: Chosen {tgt} Metal atom. {element} is added at site {site}")
            # Otherwise, coordinates are reset
            else:
               if debug >= 1: print(f"        ADD_ATOM: Chosen {tgt} Metal atom. {element} was added at site {site} but RESET due to connec={tmpconnec[posadded]}")
               isadded = False
               newlab = labels.copy()
               newcoord = coords.copy()

    return isadded, newlab, newcoord


#######################################################
def prepare_mols(
    moleclist, unique_indices, unique_species, final_charge_distribution, debug=1
):

    #############
    # SELECT: unique_indices [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2]
    # final_charge_distribution [0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0]
    #############

    Warning = False
    idxtoallocate = 0

    for idx, mol in enumerate(moleclist):

        if debug >= 1:
            print("")
        if mol.type == "Other":

            specie = unique_indices[idxtoallocate]
            spec_object = unique_species[specie][1]
            if debug >= 1:
                print(
                    "PREPARE: Molecule",
                    idx,
                    mol.labels,
                    "is specie",
                    specie,
                    "with labels",
                    spec_object.labels,
                )
            if debug >= 1:
                print("PREPARE: Molecule poscharges:", spec_object.poscharge)

            allocated = False
            if debug >= 1:
                print(
                    "PREPARE: Doing molecule", idx, "with idxtoallocate:", idxtoallocate
                )

            for jdx, ch in enumerate(spec_object.poscharge):
                if final_charge_distribution[idxtoallocate] == ch and not allocated:
                    allocated = True

                    dummy, total_charge, atom_charge, mol_object, smiles = getcharge(
                        mol.labels, mol.coord, ch, mol.conmat, mol.factor
                    )
                    mol.charge(atom_charge, total_charge, mol_object, smiles)
                    # mol.charge(spec_object.posatcharge[jdx], spec_object.poscharge[jdx], spec_object.posobjlist[jdx], spec_object.possmiles[jdx])
            if allocated:
                idxtoallocate += 1

        elif mol.type == "Complex":
            if debug >= 1:
                print(
                    "PREPARE: Molecule",
                    moleclist.index(mol),
                    "with",
                    len(mol.ligandlist),
                    "ligands",
                )

            for kdx, lig in enumerate(mol.ligandlist):
                if debug >= 1:
                    print("")
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                if debug >= 1:
                    print("PREPARE: Ligand", kdx, lig.labels, "is specie", specie)
                if debug >= 1:
                    print("PREPARE: Ligand poscharges:", spec_object.poscharge)
                allocated = False
                if debug >= 1:
                    print(
                        "PREPARE: Doing ligand",
                        kdx,
                        "with idxtoallocate:",
                        idxtoallocate,
                    )

                if lig.natoms == 2 and "N" in lig.labels and "O" in lig.labels:
                    isnitrosyl = True
                else:
                    isnitrosyl = False

                for jdx, ch in enumerate(spec_object.poscharge):
                    # if (debug >= 1): print("Doing", spec_object.poscharge, jdx, ch, final_charge_distribution[idxtoallocate])
                    # Doing [-2] 0 -2 [-2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2]
                    if final_charge_distribution[idxtoallocate] == ch and not allocated:
                        allocated = True

                        ############ RE-RUNS the Charge assignation for same-type molecules in the cell
                        #### Adds Hydrogens
                        (
                            tmplab,
                            tmpcoord,
                            addedH,
                            addedlist,
                            metal_electrons,
                        ) = define_sites(lig, mol.metalist, mol, debug)

                        tmpradii = getradii(tmplab)
                        dummy, tmpconmat, tmpconnec, tmpmconmat, tmpmconnec = getconec(tmplab, tmpcoord, lig.factor, tmpradii)

                        #### Evaluates possible charges except if the ligand is a nitrosyl
                        if isnitrosyl:
                            NO_type = get_nitrosyl_geom(lig)
                            if NO_type == "Linear":
                                NOcharge = 1
                            if NO_type == "Bent":
                                NOcharge = 0
                            dummy, dummy, uncorr_ch, mol_object, smiles = getcharge(
                                tmplab, tmpcoord, NOcharge, tmpconmat, lig.factor
                            )
                            if debug >= 1:
                                print("PREPARE: Found Nitrosyl of type=", NO_type)
                            if debug >= 1:
                                print("addedlist", addedlist)
                            if debug >= 1:
                                print("metal_electrons", metal_electrons)
                            if debug >= 1:
                                print("addedH", addedH)
                            if debug >= 1:
                                print("Sent:", NOcharge, "Obtained:", uncorr_ch)
                        else:
                            if debug >= 1:
                                print(
                                    "PREPARE: Sending getcharge with charge",
                                    ch + addedH,
                                )
                            dummy, dummy, uncorr_ch, mol_object, smiles = getcharge(
                                tmplab, tmpcoord, ch + addedH, tmpconmat, lig.factor
                            )

                        print("PREPARE: total charge obtained without correction",np.sum(uncorr_ch),"while it should be", ch+addedH)
                        print("PREPARE: smiles:", smiles)
                        #### Corrects the Charge of atoms with addedH
                        atom_charge = []
                        count = 0
                        for i, a in enumerate(
                            lig.atoms
                        ):  # Iterates over the original number of ligand atoms, thus without the added H
                            if addedlist[i] != 0:
                                count += 1
                                atom_charge.append(
                                    uncorr_ch[i]
                                    - addedlist[i]
                                    + metal_electrons[i]
                                    - uncorr_ch[lig.natoms - 1 + count]
                                )  # the last term corrects cases in which a charge is assigned to the added atom
                                # atom_charge.append(uncorr_ch[i]-addedlist[i]+metal_electrons[i])
                            else:
                                atom_charge.append(uncorr_ch[i])
                        total_charge = np.sum(atom_charge)
                        print("PREPARE: total charge obtained",total_charge,"while it should be", ch)
                        if total_charge == ch:
                            lig.charge(atom_charge, total_charge, mol_object, smiles)
                        else:
                            Warning = True

                if allocated:
                    idxtoallocate += 1

            for kdx, met in enumerate(mol.metalist):
                if debug >= 1:
                    print("")
                specie = unique_indices[idxtoallocate]
                spec_object = unique_species[specie][1]
                if debug >= 1:
                    print("PREPARE: Metal", kdx, met.label, "is specie", specie)
                if debug >= 1:
                    print("PREPARE: Metal poscharges:", spec_object.poscharge)
                allocated = False
                if debug >= 1:
                    print(
                        "PREPARE: Doing Metal",
                        kdx,
                        "with idxtoallocate:",
                        idxtoallocate,
                    )

                for jdx, ch in enumerate(spec_object.poscharge):
                    #if (debug >= 1): print("PREPARE: Checking", spec_object.poscharge, jdx, ch,"=",final_charge_distribution[idxtoallocate])
                    if final_charge_distribution[idxtoallocate] == ch and not allocated:
                        if debug >= 1:
                            print("Allocated")
                        allocated = True
                        met.charge(ch)
                if allocated:
                    idxtoallocate += 1

            if not Warning:
                ###################################################
                # Now builds the Charge Data for the final molecule
                ###################################################
                if debug >= 1:
                    print(
                        "PREPARE: Building Molecule",
                        kdx,
                        "From Ligand&Metal Information",
                    )
                tmp_atcharge = np.zeros((mol.natoms))
                tmp_smiles = []
                for lig in mol.ligandlist:
                    # print("Lig.atlist:", lig.atlist, lig.atcharge, lig.labels)
                    tmp_smiles.append(lig.smiles)
                    for kdx, a in enumerate(lig.atlist):
                        tmp_atcharge[a] = lig.atcharge[kdx]
                for met in mol.metalist:
                    tmp_atcharge[met.atlist] = met.totcharge

                mol.charge(tmp_atcharge, int(sum(tmp_atcharge)), [], tmp_smiles)

    return moleclist, Warning


#######################################################
def build_bonds(moleclist, debug=1):
    ## Builds bond data for all molecules
    ## Now that charges are known, we use the rdkit-objects with the correct charge to do that
    ## Bond entries are defined in the mol and lig objects

    # First Creates Bonds for Non-Complex Molecules
    if debug >= 1:
        print("")
    if debug >= 1:
        print("BUILD_BONDS: Doing 1st Part")
    for mol in moleclist:
        if mol.type != "Complex":
            if debug >= 1:
                print("")
            if debug >= 1:
                print("BUILD BONDS: doing mol", mol.labels, "with Natoms", mol.natoms)
            # Checks that the gmol and rdkit-mol objects have same order
            for idx, a in enumerate(mol.atoms):

                # Security Check. Confirms that the labels are the same
                if debug >= 1:
                    print("BUILD BONDS: atom", idx, a.label)
                rdkitatom = mol.object.GetAtomWithIdx(idx)
                tmp = rdkitatom.GetSymbol()
                if a.label != tmp:
                    print(
                        "Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL:",
                        a.label,
                        tmp,
                    )
                else:

                    # First part. Creates bond information
                    starts = []
                    ends = []
                    orders = []
                    for b in rdkitatom.GetBonds():
                        bond_startatom = b.GetBeginAtomIdx()
                        bond_endatom = b.GetEndAtomIdx()
                        bond_order = b.GetBondTypeAsDouble()
                        # if debug >= 1: print(bond_endatom, bond_order)
                        if (
                            mol.atoms[bond_endatom].label
                            != mol.object.GetAtomWithIdx(bond_endatom).GetSymbol()
                        ):
                            print(
                                "Error with Bond EndAtom",
                                mol.atoms[bond_endatom].label,
                                mol.object.GetAtomWithIdx(bond_endatom).GetSymbol(),
                            )
                        else:
                            if bond_endatom == idx:
                                starts.append(bond_endatom)
                                ends.append(bond_startatom)
                                orders.append(bond_order)
                            elif bond_startatom == idx:
                                starts.append(bond_startatom)
                                ends.append(bond_endatom)
                                orders.append(bond_order)
                            else:
                                print(
                                    "Warning BUILD_BONDS: Index atom is neither start nor end bond"
                                )

                    a.bonds(starts, ends, orders)

    if debug >= 1:
        print("")
    if debug >= 1:
        print("BUILD_BONDS: Doing 2nd Part")
    # 2nd Part. Creates Ligand Information
    for mol in moleclist:
        if debug >= 1:
            print("")
        if debug >= 1:
            print("BUILD BONDS: doing mol", mol.labels, "with Natoms", mol.natoms)
        if mol.type == "Complex":
            for lig in mol.ligandlist:
                if debug >= 1:
                    print("")
                if debug >= 1:
                    print(
                        "BUILD BONDS: doing ligand",
                        lig.labels,
                        "with Natoms",
                        lig.natoms,
                    )

                for idx, a in enumerate(lig.atoms):
                    # if debug >= 1: print(len(lig.atoms), lig.natoms)
                    # Security Check. Confirms that the labels are the same
                    rdkitatom = lig.object.GetAtomWithIdx(idx)
                    tmp = rdkitatom.GetSymbol()
                    if a.label != tmp:
                        print(
                            "Error in Build_Bonds. Atom labels do not coincide. GMOL vs. MOL:",
                            a.label,
                            tmp,
                        )
                        print("DEBUG:")
                        print("Ligand;", lig.labels)
                        print("Atoms of RDKIT-Object")
                        for kdx, a in enumerate(lig.object.GetAtoms()):
                            print(kdx, a.GetSymbol())
                        print("Atoms of GMOL-Object")
                        for kdx, a in enumerate(lig.atoms):
                            print(kdx, a.label)
                    else:

                        # First part. Creates bond information
                        starts = []
                        ends = []
                        orders = []
                        # if debug >= 1: print("checking bonds for atom:", a.label, idx)
                        for b in rdkitatom.GetBonds():
                            bond_startatom = b.GetBeginAtomIdx()
                            bond_endatom = b.GetEndAtomIdx()
                            bond_order = b.GetBondTypeAsDouble()
                            # if debug >= 1: print(bond_startatom, bond_endatom, bond_order, lig.natoms)
                            if (
                                bond_startatom >= lig.natoms
                                or bond_endatom >= lig.natoms
                            ):
                                continue
                                # if debug >= 1: print("Found a mention to a dummy H atom")
                            else:
                                # if debug >= 1: print(lig.atoms[bond_endatom].label, lig.object.GetAtomWithIdx(bond_endatom).GetSymbol())
                                if (
                                    lig.atoms[bond_endatom].label
                                    != lig.object.GetAtomWithIdx(
                                        bond_endatom
                                    ).GetSymbol()
                                ):
                                    print(
                                        "Error with Bond EndAtom",
                                        lig.atoms[bond_endatom].label,
                                        lig.object.GetAtomWithIdx(
                                            bond_endatom
                                        ).GetSymbol(),
                                    )
                                else:
                                    if bond_endatom == idx:
                                        starts.append(bond_endatom)
                                        ends.append(bond_startatom)
                                        orders.append(bond_order)
                                    elif bond_startatom == idx:
                                        starts.append(bond_startatom)
                                        ends.append(bond_endatom)
                                        orders.append(bond_order)
                        #                                     starts.append(bond_startatom)
                        #                                     ends.append(bond_endatom)
                        #                                     orders.append(bond_order)

                        a.bonds(starts, ends, orders)

    if debug >= 1:
        print("BUILD_BONDS: Doing 3rd Part")
    # 3rd Part. Merges Ligand Information into Molecule Object using the atlists
    for mol in moleclist:
        if mol.type == "Complex":
            allstarts = []
            allends = []
            allorders = []
            # Adds atoms within ligands
            for lig in mol.ligandlist:
                for a in lig.atoms:
                    for b in a.bond:
                        allstarts.append(lig.atlist[b[0]])
                        allends.append(lig.atlist[b[1]])
                        allorders.append(b[2])

            # Adds Metal-Ligand Bonds, with an arbitrary 0.5 order:
            for idx, row in enumerate(mol.mconmat):
                # if debug >= 1: print(row)
                for jdx, val in enumerate(row):
                    if val > 0:
                        # if debug >= 1: print(idx, jdx, val)
                        allstarts.append(idx)
                        allends.append(jdx)
                        allorders.append(0.5)

            # I sould work to add Metal-Metal Bonds. Would need to work on the Metal class:

            # Finally, puts everything together, and creates bonds for MOLECULE atom objects
            for idx, a in enumerate(mol.atoms):
                starts = []
                ends = []
                orders = []
                group = []
                for entry in zip(allstarts, allends, allorders):
                    if entry[0] == idx or entry[1] == idx:
                        if (
                            entry not in group
                            and (entry[1], entry[0], entry[2]) not in group
                        ):
                            starts.append(entry[0])
                            ends.append(entry[1])
                            orders.append(entry[2])
                            group.append(entry)

                a.bonds(starts, ends, orders)

    return moleclist


#######################################################
def check_carbenes(atom, ligand, molecule):
    # Function that determines whether a given connected "atom" of a "ligand" of a "molecule" is a carbene
    # This function is in progress. Ideally, should be able to identify Fischer, Schrock and N-Heterocyclic Carbenes
    # The former two cases probably require rules that involve other ligands in the molecule, hence why the "molecule" is provided
    #:return iscarbene: Boolean variable. True/False
    #:return element:   Type of element that will be later added in the "add_atom" function below
    #:return addedlist: List of integers which track in which atom of the ligand we're adding "elements"
    #:return metal_electrons: List of integers, similar to addedlist, which track in which atom of the ligand we're counting on metal_electrons.

    # about Metal electrons: This variable is a way to contemplate cases in which the metal atom is actually contributing with electrons to the metal-ligand bond.
    # about Metal electrons: In reality, I'm not sure about how to use it correctly, and now is used without much chemical sense

    iscarbene = False
    element = "H"
    addedlist = 0
    metal_electrons = 0

    # Initial attempt with Carbenes, but they are much more complex
    # Looks for Neighbouring N atoms
    list_of_coord_atoms = []
    for i in atom.adjacency:
        list_of_coord_atoms.append(ligand.labels[i])
    numN = list_of_coord_atoms.count("N")

    if numN == 2:  # it is an N-Heterocyclic carbenes
        iscarbene = True
        element = "H"
        addedlist = 1

    return iscarbene, element, addedlist, metal_electrons
