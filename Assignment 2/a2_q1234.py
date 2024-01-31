
# Add the code for this function as described in the notebook 
# def evaluate(s):
 

def evaluate(expression):
    parts = expression.split()
    
    if parts[0] == '&':
        result = int(parts[1]) and int(parts[2])
    elif parts[0] == '|':
        result = int(parts[1]) or int(parts[2])
    elif parts[0] == '=>':
        result = (int(parts[1]) == 0) or int(parts[2])
    elif parts[0] == '<=>':
        result = int(parts[1]) == int(parts[2])

    return result




def evaluate_with_bindings(expression, bindings):
    def evaluate_symbol(symbol):
        if symbol.startswith('~'):
            return not bindings.get(symbol[1:], False)
        return bindings.get(symbol, False)

    parts = expression.split()
    operator = parts[0]
    operand1 = parts[1]
    operand2 = parts[2]

    if operand1.isdigit():
        operand1 = int(operand1)
    else:
        operand1 = evaluate_symbol(operand1)

    if operand2.isdigit():
        operand2 = int(operand2)
    else:
        operand2 = evaluate_symbol(operand2)

    if operator == '&':
        result = operand1 and operand2
    elif operator == '|':
        result = operand1 or operand2
    elif operator == '=>':
        result = (not operand1) or operand2
    elif operator == '<=>':
        result = operand1 == operand2

    return result


# Examples test cases
e1 = "| 0 1"
e2 = "<=> 1 1"
e3 = "& 0 0"

res_e1 = evaluate(e1)
res_e2 = evaluate(e2)
res_e3 = evaluate(e3)


print(f'{e1} = {res_e1}')
print(f'{e2} = {res_e2}')
print(f'{e3} = {res_e3}')

d = {'foo': 0, 'b': 1}
print(d)
be1 = '& 0 1'
be2 = '& foo 1'
be3 = '& foo ~b'


# Add the code for this function 
# def evaluate_with_bindings(s,d):
  

# Example test cases 
res_be1 = evaluate_with_bindings(be1,d)
res_be2 = evaluate_with_bindings(be2,d)
res_be3 = evaluate_with_bindings(be3,d)

print(f'{be1} = {res_be1}')
print(f'{be2} = {res_be2}')
print(f'{be3} = {res_be3}')


# Add the code for this function as described in the notebook 
# You can add helper functions if you want as long as the function works as expected 

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

def prefix_eval(input_str, bindings):
    input_list = input_str.split(' ')
    result, _ = recursive_logic_eval(input_list, bindings)
    return result

d = {"a": 1, "b": 0}
pe1 = "& a | 0 1"
pe2 = "& 0 | 1 b"
pe2 = "| 1 => ~b b"
pe3 = "<=> b <=> ~b 0"
pe4 = "=> 1 & a 0"
pe5 = "& ~a <=> 0 0"

print(d)
for e in [pe1,pe2,pe3,pe4,pe5]:
    print("%s \t = %d" % (e, prefix_eval(e,d)))

### SAMPLE OUTPUT 
# | 0 1 = 1
# <=> 1 1 = 1
# & 0 0 = 0
# {'foo': 0, 'b': 1}
# & 0 1 = 0
# & foo 1 = 0
# & foo ~b = 0
# {'a': 1, 'b': 0}
# & a | 0 1        = 1
# | 1 => ~b b      = 1
# <=> b <=> ~b 0   = 1
# => 1 & a 0       = 0
# & ~a <=> 0 0     = 0



