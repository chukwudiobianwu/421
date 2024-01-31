# YOUR CODE GOES HERE 
def recursive_logic_eval(tokens, bindings):
    head, tail = tokens[0], tokens[1:]

    if head in ['&', '|', '=>', '<=>']:
        val1, tail = recursive_logic_eval(tail, bindings)
        val2, tail = recursive_logic_eval(tail, bindings)

        if head == '&':
            return int(val1) and int(val2), tail
        elif head == '|':
            return int(val1) or int(val2), tail
        elif head == '=>':
            return int(not val1) or int(val2), tail
        elif head == '<=>':
            return int(val1 == val2), tail
    elif head.startswith('~'):
        variable = head[1:]
        negated_value = not bindings.get(variable, False)
        return int(negated_value), tail
    elif head.isdigit():
        return int(head), tail
    else:
        variable_value = bindings.get(head, False)
        return int(variable_value), tail

def prefix_logic_eval(input_str, bindings):
    input_list = input_str.split(' ')
    result, _ = recursive_logic_eval(input_list, bindings)
    return result


# YOUR CODE GOES HERE 
def generate_all_models(variables):
    num_variables = len(variables)
    max_value = 2 ** num_variables
    models = []

    for i in range(max_value):
        model = {var: (i >> j) & 1 for j, var in enumerate(variables)}
        models.append(model)

    return models

def check_validation(result,result2):

    # Check the result
    if result and result2:
        print("\nkB DOES ENTAIL A.")
    else:
        print("\nKB DOES NOT ENTAIL ALPHA.")

def entails_using_model_checking(kb, alpha, bindings):

    
    all_models = generate_all_models(bindings)
    Set = False
    Set2 = True
    for i in all_models:
        KB = prefix_logic_eval(kb, i)
        A = prefix_logic_eval(alpha, i)
        
        if KB and A:
            print(f"Name: Model {i}, \nKB: {KB}, alpha: {A}")
            Set = True
        elif (KB == 1 and A == 0):
            print(f"Name: Model {i}, \nKB: {KB}, alpha: {A}")
            Set2 = False
        else:
            print(f"Name: Model {i}, \nKB: {KB}, alpha: {A}")
            
    check_validation(Set,Set2)
            

# Sample variable bindings
variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Sample KB and alpha
kb = "& A & | B C & D & E & ~F ~G"
alpha = "& A & D & E & ~F ~G"

entails_using_model_checking(kb, alpha, variables)
