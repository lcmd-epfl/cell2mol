# To maximize python3/python2 compatibility
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

# The unicode type does not exist in Python3 as the str type
# encompasses unicode.  PyCIFRW tests for 'unicode' would fail
# Suggestions for a better approach welcome.

if isinstance(u"abc",str):   #Python3
    unicode = str

import re
from CifFile import CifError
pycifrw_func_table = {   #how to use PyCIFRW CifFile objects
         "data_access": "ciffile[%s]",   # argument is dataname to be accessed
         "optional_data_access":"ciffile.get(%s,None)",
         "element_no": "%s[%s]", # accessing a particular element of the result of data_access
         "count_data": "len(%s)", # number of elements for result of data_access
         "cat_names":"ciffile.dictionary.names_in_cat(%s)", #names in category %s
         "has_name":"ciffile.has_key_or_alias(%s)",
         "semantic_packet":"ciffile.GetKeyedSemanticPacket(%s,%s)" #get a packet for key, value
                     }

def make_python_function(in_ast,func_name,targetname, special_ids=[{}],
                         func_table = pycifrw_func_table, cif_dic=None,cat_meth=False, 
                         func_def = False, have_sn=True,loopable={},debug=None,depends=False):
    """Convert in_ast to python code"""
    if debug is not None:
        print("incoming AST:")
        print(repr(in_ast))
    func_text,withtable,dependencies,cur_row = start_traverse(in_ast,func_table,target_id=targetname,cat_meth=cat_meth,loopable=loopable,debug=debug,func=func_def,cif_dic=cif_dic)
    if debug is not None:
        print('Start========')
        print(func_text)
        print('End==========')
    if func_def and not depends:
        return func_text
    elif func_def:
        return func_text, None
    # now indent the string
    noindent = func_text.splitlines()
    # get the minimum indent and remove empty lines
    no_spaces = [re.match(r' *',a).end() for a in noindent if a] #drop empty lines
    min_spaces = min(no_spaces)+4   # because we add 4 ourselves to everything
    if len(withtable) > 0 or cur_row:  # a loop method
            d_vars = [a[1][0] for a in withtable.items()]
            if cur_row:
                d_vars = d_vars + ['__current_row']
            dummy_vars = ",".join(d_vars)
            actual_names = [k for k in withtable.keys()]
            actual_names = [func_table["data_access"]% ("'"+a+"'") for a in actual_names]
            # intercept optional values and replace with [] if None
            optional_names = ["__option_w%d"%n for n,k in enumerate(withtable.keys()) if withtable[k][2]]
            if cur_row:
                actual_names+=['__row_id']
            final_names = actual_names[:]
            one_pack_names = [func_table["element_no"] % (a,"packet_no") for a in actual_names]
            for n,k in enumerate(withtable.keys()):
                if withtable[k][2]:
                    final_names[n]="__option_w%d"%n  #pre-evaluated
                    one_pack_names[n] = "__option_w%d"%n
            map_names = ",".join(final_names)
            one_packet_each = ",".join(one_pack_names)
            preamble = "def %s(ciffile,packet_no=-1):\n" % (func_name)
            preamble +="    try:\n"
            preamble +="        from itertools import repeat,imap\n"   #note that imap might fail
            preamble +="    except ImportError: #python3\n"
            preamble +="        imap = map\n"
            preamble +="    def drel_func(%s):\n" % dummy_vars
            # for debugging
            print_instruction = "'Function passed variables "+("{!r} "*len(d_vars))+"'.format("+dummy_vars+",)"
            preamble +="        print(%s)\n" % print_instruction
            # preamble +="        print('Globals inside looped drel_func:' + repr(globals())\n")
            # the actual function gets inserted here
            # Handle the optional names
            end_body = "\n"
            for n,one_opt in enumerate(withtable.keys()):
                if withtable[one_opt][2]:
                    end_body += "    try:\n"
                    end_body += "            %s%d = %s\n" % ("__option_w",n,func_table["optional_data_access"]%("'"+one_opt+"'"))
                    end_body += "    except KeyError:\n"
                    end_body += "            %s%d = None\n" % ("__option_w",n)
            end_body+=      "    if packet_no < 0:   #map\n"
            for one_opt in optional_names:
                end_body += "        if %s is None: %s = repeat(None)\n" % (one_opt,one_opt)

            if cur_row and len(actual_names) > 1:    #i.e. have real names from category
                end_body +="        __row_id = range(%s)\n" % (func_table['count_data'] % ("'"+actual_names[0]+"'"))
            elif cur_row and len(actual_names)==1:   #so no actual names available

                end_body += "        cat_names = %s\n" % (func_table["cat_names"] % ("'"+getcatname(targetname)[0]+"'"))
                end_body += "        have_name = [a for a in cat_names if %s]\n" % (func_table["has_name"] % "a")
                end_body += "        if len(have_name)>0:\n"
                end_body += "           full_length = %s \n" % (func_table["count_data"] % (func_table["data_access"] % "have_name[0]"))
                end_body += "           __row_id = range(full_length)\n"
                end_body += "        else:\n"
                end_body += "           return []\n"

            end_body+=      "        return list(imap(drel_func,%s))\n" % (map_names+",")
            end_body+=     "    else:\n"
            end_body+=     "        return drel_func(%s)\n" % one_packet_each

    else:
            preamble = "def %s(ciffile):\n" % func_name
            #preamble +="        global StarList#from CifFile.drel import drel_runtime\n"
            end_body = ""

    if cat_meth:
        preamble += " "*8 + "__dreltarget = {}\n" # initialise
    num_header = """
        import math,cmath
        try:
            import numpy
        except:
            print("Can't import numerical python, this method may not work")
"""
    preamble += num_header
    indented = map(lambda a:" "*8 + a +"\n",noindent)  #indent dREL body
    postamble = ""
    postamble += " "*8 + "return __dreltarget"
    final = preamble + "".join(indented) + postamble + end_body
    if not depends:
        return final
    else: return final, dependencies

def start_traverse(in_node,api_table,target_id=None,loopable={},cat_meth=False,debug=None, func=False,
                  cif_dic=None):
  special_info = {"special_id":[{}],"target_id":target_id,"withtable":{},"sub_subject":"",
                  "depends":set(),"loopable_cats":loopable,"packet_vars":{},
                  "need_current_row":False,"rhs":None,"inif":False}
  # create a virtual enclosing 'with' statement
  if target_id is not None and not cat_meth and not func:
      cat,name = getcatname(target_id)
      special_info["special_id"][-1].update({"_"+cat:[cat,"",False]})
      if cat in special_info["loopable_cats"].keys():    #
          special_info["special_id"][-1]["_"+cat][1] = "looped_cat"
  mathop_table = {"+":None, "-":None, "<":"<", "*":None, "/":None,
                  "&":"&", "|":"|",
                  ">":">", "<=":"<=", ">=":">=", "!=":"!=",
                  "or":" or ", "and":" and ",
                  "==":"==", "in":" in ", "not in":" not in ",
                  "^":None,"**":"**"}

  aug_assign_table = {"++=":"drel_runtime.aug_append",
                      "+=":"drel_runtime.aug_add",
                      "-=":"drel_runtime.aug_sub",
                      "--=":"drel_runtime.aug_remove"}

  def traverse_ast(in_node,debug=debug):
    if isinstance(in_node,(unicode,str)):
        return in_node
    if isinstance(in_node[0],list):
        raise SyntaxError('First element of AST Node must be string: ' + repr(in_node))
    node_type = in_node[0]
    if debug == node_type:
        print(node_type + ": " + repr(in_node))
    if node_type == "ARGLIST":
        pass
    elif node_type == "BINARY":
        return("%d" % int(in_node[1],base=2))
    elif node_type == "FALSE":
        return("False")
    elif node_type == "REAL":
        return(in_node[1])
    elif node_type == "HEX":
        return("%d" % int(in_node[1],base=16))
    elif node_type == "INT":
        return(in_node[1])
    elif node_type == "IMAGINARY":
        return(in_node[1])
    elif node_type == "OCTAL":
        return("%d" % int(in_node[1],base=8))

    elif node_type == "ATOM":
        if isinstance(in_node[1],(unicode,str)):
            # pick up built-in literals
            if in_node[1].lower() == 'twopi':
                return "(2.0 * math.pi)"
            if in_node[1].lower() == 'pi':
                return "math.pi"
            else:
                return in_node[1]
        else:
            return traverse_ast(in_node[1])
    elif node_type == "ITEM_TAG":
        return in_node[1]
    elif node_type == "LITERAL":
        return in_node[1]
    elif node_type == "LIST":
        if len(in_node)==1:  #empty list
           return "StarList([])"
        if special_info["rhs"] == True:
            outstring = "StarList(["
        else:
            outstring = ""
        for list_elem in in_node[1:]:
            outstring = outstring + traverse_ast(list_elem) + ","
        if special_info["rhs"] == True:
            return outstring[:-1] + "])"
        else:
            return outstring[:-1]
    elif node_type == "TABLE":
        if len(in_node)==1:
            return "StarTable({})"
        else:
            outstring = "{"
        for table_elem in in_node[1:]:
            outstring = outstring + traverse_ast(table_elem[0])+":"+traverse_ast(table_elem[1]) +","
        return outstring[:-1] + "}"
    elif node_type == "SUBSCRIPTION":  # variable, single expression
        newid = 0
        if in_node[1][0] == "ATOM" and in_node[1][1][0] == "ITEM_TAG":  #keyed lookup
            print("Found category used as item tag: subscribing")
            newid = [in_node[1][1][1][1:],False,False]  #drop underscore and remember
        else:
            primary = traverse_ast(in_node[1])
            # check to see if this is a special variable
            for idtable in special_info["special_id"]:
                newid = idtable.get(primary,0)
                if newid: break
                if primary in special_info["loopable_cats"].keys():   #loop category used
                    newid = [primary,False,False]
                    break
        if newid:
            #FIXME: the dataname may not be the <cat>.<obj> construction (eg pdCIF)
            key_items = ["_"+newid[0]+"."+s for s in special_info["loopable_cats"][newid[0]][0]]  #key name
            special_info["depends"].update([k.lower() for k in key_items])
            get_loop = api_table["semantic_packet"] % (traverse_ast(in_node[2]),"'"+newid[0]+"'")
            special_info["sub_subject"] = newid[0]  #in case of attribute reference following
            print("Set sub_subject to %s" % special_info["sub_subject"])
            return get_loop
        else:
            outstring = primary + "["
            outstring = outstring + traverse_ast(in_node[2]) + "]"
            return outstring

    elif node_type == "ATTRIBUTE":  # id/tag , att
        outstring = ""
        newid = 0
        # check for special ids
        primary = traverse_ast(in_node[1])  # this will set sub_subject if necessary
        for idtable in special_info["special_id"]:
            newid = idtable.get(primary,0)
            if newid: break
        if newid:
            #catch our output name
            true_name = cif_dic.get_name_by_cat_obj(newid[0].lower(),in_node[2].lower()).lower()
            if true_name == special_info.get("target_id","").lower():
                    outstring = "__dreltarget"
                    special_info["have_drel_target"] = True
            # if we are looping, we add a loop prefix. If we are withing an
            # unlooped category, we put the full name back.
            elif newid[2] or (not newid[2] and not newid[1]):   # looping or simple with
                outstring = api_table["data_access"] % ('"' +true_name +'"')
                special_info["depends"].add(true_name)
                if newid[1]:  # a loop statement requires an index
                    outstring += "[" + newid[1]+ "]"
            else:   # a with statement; capture the name and create a dummy variable
                if true_name not in special_info["withtable"]:  #new
                    position = len(special_info["withtable"])
                    new_var = "__w%d" % position
                    isoptional = special_info["inif"]
                    special_info["withtable"][true_name] = (new_var,position,isoptional)
                outstring += special_info["withtable"][true_name][0]
                special_info["depends"].add(true_name)
        elif in_node[1][0] == "ATOM" and primary[0] == "_":   # a cat/obj name
            fullname = cif_dic.get_name_by_cat_obj(primary,in_node[2]).lower()
            # a simple cat.obj dataname from the dictionary
            if special_info.get("target_id","").lower() == fullname:
                outstring = "__dreltarget"
                special_info["have_drel_target"] = True
            else:
                special_info["depends"].add(fullname)
                outstring = api_table["data_access"] % ("'" + fullname + "'")
        else: # default to Python attribute access
            # check for packet variables
            if primary in special_info["packet_vars"]:
                real_cat = special_info["packet_vars"][primary]
                fullname = cif_dic.get_name_by_cat_obj(real_cat,in_node[2])
                special_info['depends'].add(fullname)
            elif special_info["sub_subject"]:
                fullname = cif_dic.get_name_by_cat_obj(special_info["sub_subject"],in_node[2])
                special_info['depends'].add(fullname)
            else:  # not anything special
                fullname = in_node[2]
            outstring = "getattr(" + primary + ",'" + fullname + "')"
            # sub_subject no longer relevant after attribute resolution
            special_info['sub_subject'] = ""
        return outstring

    elif node_type == "FUNC_CALL":
        if in_node[1] == "Current_Row":  #not a function but a keyword really
            outstring = "__current_row"
            special_info["need_current_row"]=True
        else:
            func_name,every_arg_prefix,postfix = get_function_name(in_node[1])
            outstring = func_name + "( "
            if func_name == "list" and len(in_node[2])>1:   #special case
                outstring = outstring + "["
            for argument in in_node[2]:
                outstring = outstring + every_arg_prefix + traverse_ast(argument) + ","
            if postfix == None:  # signal for dictionary defined
                outstring = outstring + "ciffile)"
            else:
                outstring = outstring[:-1]
                if func_name == "list" and len(in_node[2])>1:
                    outstring = outstring + "]"
                outstring = outstring + ")" + postfix
        return outstring

    elif node_type == "SLICE":  # primary [[start,finish,step],[...]
        outstring = traverse_ast(in_node[1]) + "["
        slice_list = in_node[2]
        for one_slice in slice_list:
            if one_slice[0] == "EXPR":   #not a slice as such
                outstring += traverse_ast(one_slice)
            elif len(one_slice) == 0:
                outstring += ":"
            elif len(one_slice) >0:    # at least start
                outstring += traverse_ast(one_slice[0]) + ":"
                if len(one_slice) >1:    #start,finish only
                    outstring += traverse_ast(one_slice[1])
                if len(one_slice) == 3:    #step as well
                    outstring += ":" + traverse_ast(one_slice[2])
            outstring += ","
        outstring = outstring[:-1] + "]"
        return outstring

    elif node_type == "MATHOP":
        op = mathop_table[in_node[1]]
        first_arg = traverse_ast(in_node[2])
        second_arg = traverse_ast(in_node[3])
        if op is not None:    #simple operation
            outstring = first_arg + op + second_arg
        else:
            outstring = fix_mathops(in_node[1],first_arg,second_arg)
        return outstring
    elif node_type == "SIGN":
        outstring = "drel_runtime.drel_dot(" + in_node[1] + "1," + traverse_ast(in_node[2])+")"
        return outstring
    elif node_type == "UNARY":
        outstring = in_node[1] + " " + traverse_ast(in_node[2])
        return outstring

    elif node_type == "IF_EXPR":   #IF_EXPR test true_suite [ELSE IF_EXPR] false_suite
        outstring = "if "
        outstring = outstring + traverse_ast(in_node[1])
        outstring = outstring + ":"
        old_inif = special_info["inif"]
        special_info["inif"] = True
        true_bit = traverse_ast(in_node[2])
        outstring = outstring + add_indent("\n"+true_bit)  #indent
        elseif = in_node[3]
        if len(elseif)!=0:
            for one_cond in elseif:  #each entry is condition, suite
                outstring += "\nelif " + traverse_ast(one_cond[0]) + ":"
                outstring += add_indent("\n" + traverse_ast(one_cond[1]))
        if len(in_node)>4:
            outstring = outstring + "\nelse:"
            false_bit = traverse_ast(in_node[4])
            outstring = outstring + add_indent("\n"+false_bit)  #indent
        special_info["inif"] = old_inif
        return outstring

# dREL for statements include the final value, whereas a python range will include
# everything up to the final number
    elif node_type == "DO":    #DO ID = start, finish, incr, suite
        outstring = "for " + in_node[1] + " in range(" + traverse_ast(in_node[2]) + ","
        finish = traverse_ast(in_node[3])
        increment = traverse_ast(in_node[4])
        outstring = outstring + finish + "+1" + "," + increment
        outstring = outstring + "):"
        suite = add_indent("\n"+traverse_ast(in_node[5]))
        return outstring + suite
    elif node_type == "FOR": # FOR target_list expression_list suite
        outstring = "for "
        for express in in_node[1]:
            outstring = outstring + traverse_ast(express) + ","
        outstring = outstring[:-1] + " in "
        special_info["rhs"] = True
        for target in in_node[2]:
            outstring += "copy("+traverse_ast(target) + "),"
        special_info["rhs"] = None
        outstring = outstring[:-1] + ":" + add_indent("\n" + traverse_ast(in_node[3]))
        return outstring
    elif node_type == "REPEAT": #REPEAT suite
        outstring = "while True:" + add_indent("\n" + traverse_ast(in_node[1]))
        return outstring
    elif node_type == "WITH": #new_id old_id suite
        # each entry in special_id is [alias:[cat_name,loop variable, is_loop]]
        alias_id = in_node[1]
        cat_id = in_node[2]
        is_already_there = [a for a in special_info['special_id'][-1].keys() if \
            special_info['special_id'][-1][a][0] == cat_id]
        if len(is_already_there)>0:
            del special_info['special_id'][-1][is_already_there[0]]
            print("Found explicit loop category alias: %s for %s" % (alias_id,cat_id) )
        special_info['special_id'][-1].update({alias_id:[cat_id,"",False]})
        if in_node[2] in special_info['loopable_cats'].keys(): #flag this
            special_info['special_id'][-1][alias_id][1] = "looped_cat"
        outstring = traverse_ast(in_node[3])
        return outstring

    elif node_type == "LOOP": #ALIAS CAT LOOPVAR COMP COMPVAR SUITE 
        alias_id = in_node[1]
        cat_id = in_node[2]
        var_info = [cat_id,"",False]
        if cat_id not in special_info['loopable_cats'].keys():
            message =  "%s is not a loopable category (must be one of:\n%s)" % (cat_id,special_info['loopable_cats'].keys())
            print(message)
            raise CifError(message)
        #loop over some index
        loop_num = len(special_info['special_id'][-1])+1
        if in_node[3] == "":  # provide our own
            loop_index = "__pi%d" % loop_num
        else:
            loop_index = in_node[3]
        var_info[1] = loop_index
        var_info[2] = True
        special_info['special_id'][-1].update({alias_id:var_info})
        # now emit some text: first to find the length of the category
        # loopable cats contains a list of names defined for the category
        # this might not be robust as we ignore alternative resolutions of the (cat,name) pair
        catnames = set([a[1][0] for a in cif_dic.cat_obj_lookup_table.items() if a[0][0]==cat_id.lower()])
        outstring = "__pyallitems = " + repr(catnames)
        outstring += "\nprint('names in cat = %s' % repr(__pyallitems))"
        outstring += "\n" + "__pycitems = [a for a in __pyallitems if %s]" % (api_table["has_name"] % "a")
        outstring += "\nprint('names in cat -> %s' % repr(__pycitems))\n"
        cat_key = cif_dic[cat_id]['_category_key.name'][0]   #take official key
        # If there is nothing in the category, provoke category creation by evaluating the key
        outstring += "if len(__pycitems)==0:\n"
        outstring += "    __pydummy = %s\n" % (api_table["data_access"] % repr(cat_key))
        outstring += "    __pycitems = [a for a in __pyallitems if %s]\n" % (api_table["has_name"] % "a")
        outstring += "    print('After category creation, names in cat ->' + repr(__pycitems))\n"
        special_info["depends"].add(cat_key)  #add key as a dependency
        if var_info[2] == True:
            access_string = api_table["count_data"] % (api_table["data_access"] % "__pycitems[0]")
            outstring += "\n" + "__loop_range%d = range(%s)" % (loop_num,access_string)
        else:
            outstring += "\n" + "__loop_range%d = [0]" % loop_num
            #outstring +="\n" + "for __noloop in [0]:"
        # deal with this comparison test
        if in_node[4] != "":
            outstring += "\n" + "__loop_range%d = [a for a in __loop_range%d if a %s %s]" % (loop_num,loop_num,in_node[4],in_node[5])
        # now output the looping command
        outstring += "\n" + "for %s in __loop_range%d:" % (loop_index,loop_num)
        # now the actual body of the loop
        loop_body = traverse_ast(in_node[6])
        outstring = outstring + add_indent("\n"+loop_body)
        return outstring

    elif node_type == "FUNCTION":   #FUNCTION ID ARGLIST SUITE
        func_name = in_node[1]
        outstring = "def %s (" % func_name
        for one_arg in in_node[2]:
            outstring += one_arg[0] + ","
        outstring = outstring + "ciffile):"
        # imports
        #import_lines = "import numpy\nfrom CifFile.drel import drel_runtime\n"
        import_lines = ""
        outstring = outstring + add_indent("\n" + import_lines + traverse_ast(in_node[3])+"\nreturn %s" % func_name)
        return outstring


    elif node_type == "STATEMENTS":
        outstring = ""
        for one_statement in in_node[1]:
#            try:
                next_bit = traverse_ast(one_statement)
                if not isinstance(next_bit,(unicode,str)):
                    print("Unable to traverse AST for %s" % one_statement[0])
                else:
                    outstring = outstring + next_bit + "\n"
#            except SyntaxError as message:
#                print("Failed, so far have \n " + outstring)
#                outstring += "raise SyntaxError, %s" % message
#            except:
#                print("Failed, so far have \n " + outstring)
#                outstring += "raise SyntaxError, %s" % `one_statement`
        return outstring
    elif node_type == "ASSIGN":  #Target_list ,assigner, expression list
        outstring = ""
        lhs_values = []
        special_info["rhs"] = False
        for target_value in in_node[1]:
            one_value = traverse_ast(target_value)
            outstring = outstring + one_value +","
            lhs_values.append(one_value)
        lhs = outstring[:-1]
        rhs = ""
        special_info["rhs"] = True
        for order,expression in enumerate(in_node[3]):
            rhs += traverse_ast(expression)+","
            if special_info["sub_subject"] != "":   #a full packet
                special_info["packet_vars"].update({lhs_values[order]:special_info["sub_subject"]})
                special_info["sub_subject"] = ""
        # we cannot expand a numpy array, hence the workaround here
        #if in_node[2] == "++=":
        #    outstring = "_temp1 = %s;%s = %s(_temp1,%s)" % (lhs,lhs,aug_assign_table["++="],rhs[:-1])
        if in_node[2] != "=":
            outstring = "%s = %s(%s,%s)" % (lhs, aug_assign_table[in_node[2]],lhs,rhs[:-1])
        else:
            outstring = "%s = %s" % (lhs,rhs[:-1])
        special_info["rhs"] = None
        return outstring
    elif node_type == "FANCY_ASSIGN":  # [1] is cat name, [2] is list of objects
        catname = in_node[1]
        outstring = ""
        special_info["rhs"] = True
        for obj,value in in_node[2]:
            real_id = cif_dic.get_name_by_cat_obj(catname, obj)
            newvalue = traverse_ast(value)
            outstring = outstring + "__dreltarget.update({'%s':__dreltarget.get('%s',[])+[%s]})\n" % (real_id,real_id,newvalue)
        special_info["rhs"] = None
        return outstring

    elif node_type == "LIST":
        outstring = "["
        for one_element in in_node[1]:
            outstring = outstring + traverse_ast(one_element)  + ","
        return outstring + "]"
    elif node_type == "EXPR":
        return traverse_ast(in_node[1])
    # Expr list occurs only when a non-assignment statement appears as expr_stmt
    elif node_type == "EXPRLIST":
        outstring = ""
        for one_expr in in_node[1]:
            outstring += traverse_ast(one_expr) + "\n"
        return outstring
    elif node_type == "GROUP":
        outstring = "("
        for expression in in_node[1]:
             outstring = outstring + traverse_ast(expression) + ","
        return outstring[:-1] + ")"
    elif node_type == "PRINT":
        return 'print( ' + traverse_ast(in_node[1]) + ")"
    elif node_type == "BREAK":
        return 'break '
    elif node_type == "NEXT":
        return 'continue '

    else:
       return "Not found: %s" % repr(in_node)
  result = traverse_ast(in_node)
  # remove target id from dependencies
  if special_info["target_id"] is not None:
      special_info["depends"].discard(special_info["target_id"].lower())
  if not special_info.get("have_drel_target",False):
      print('WARNING: no assignment to __dreltarget in %s (this is OK for category methods)' % repr(target_id))
      print(result)
  return result,special_info["withtable"],special_info["depends"],special_info["need_current_row"]

def get_function_name(in_name):
    """Return the Python name of the dREL function, an argument prefix,
       and anything to be appended to the end"""
    builtins = {"table":"dict",
                "list":"list",
                "array":"numpy.array",
                "len":"len",
                "abs":"abs",
                "magn":"abs",
                "atoi":"int",
                "float":"float",
                "str":"str",
                "array":"numpy.array",
                "norm":"numpy.linalg.norm",
                "sqrt":"math.sqrt",
                "exp":"math.exp",
                "complex":"complex",
                "max":"max",
                "min":"min",
                "strip":"drel_runtime.drel_strip",
                "int":"drel_runtime.drel_int",
                "eigen":"drel_runtime.drel_eigen",
                "hash":"hash"  #dREL extension
    }
    test_name = in_name.lower()
    target_name = builtins.get(test_name,None)
    if target_name is not None:
        return target_name,"",""
    if test_name in ['sind','cosd','tand']:
        return "math."+test_name[:-1],"math.radians(",")"
    if test_name in ['acosd','asind','atand','atan2d']:
        return "math.degrees(math."+test_name[:-1],"",")"
    if test_name == "mod":
        return "divmod","","[1]"
    if test_name == "upper":
        return "","",".upper()"
    if test_name == "transpose":
        return "","",".T"
    if test_name == 'expimag':
        return "cmath.exp","1j*(",")"
    if test_name in ['real','imag']:
        return "","","." + test_name
    if test_name == 'matrix':
        return "numpy.matrix","",".astype('float64')"
    if test_name == 'sort':
        return "","",".sort()"
    return in_name,"",None   #dictionary defined

def fix_mathops(op,first_arg,second_arg):
    """Return a string that will carry out the requested operation"""
    if op == "^":
        return "numpy.cross(%s,%s)" % (first_arg,second_arg)
    elif op == "*":  #could be matrix multiplication
        return "drel_runtime.drel_dot(%s,%s)" % (first_arg,second_arg)
    elif op == "+":
        return "drel_runtime.drel_add(%s,%s)" % (first_arg, second_arg)
    elif op == "-":
        return "numpy.subtract(%s,%s)" % (first_arg, second_arg)
    # beware integer division on this one...
    elif op == "/":
        return "numpy.true_divide(%s,%s)" % (first_arg, second_arg)

def add_indent(text,n=4):
    """Indent text by n spaces"""
    return re.sub("\n","\n"+4*" ",text)

def getcatname(dataname):
    """Return cat,name pair from dataname"""
    try:
        cat,name = dataname.split(".")
    except ValueError:        #no period in name
        return cat,None
    return cat[1:],name

