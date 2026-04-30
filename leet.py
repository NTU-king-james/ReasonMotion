class Solution:
    def calculate(self, s: str) -> int:

        # contain: digit, +, -, (

        if s[0] == '(':
            i = 1
        if s[0] == '-':
            neg = True
            if s[0] == '(':
            i = 1
        else:
            neg = False
            i = 0
        
        return -self.recursive(s[i]) if neg else self.recursive(s[i])
        
    def recursive(self, s: str) -> int:

        stack = []
        operands = ['+', '-']
        while(i < len(s) and s[i] != ')'):
            if s[i] == ' ':
                i += 1
                continue
            elif s[i] in operands:
                fst = stack.pop()
                if s[i+1] == '(': 
                    sec = self.calculate(s[i+2])
                    while(s[i] != ')'):
                        i += 1
                else: sec = s[i+1]
                if s[i] == '+': stack.push(sec + fst)
                elif s[i] == '-': stack.push(sec - fst)
            elif s[i] == '(':
                stack.pop(sec = self.calculate(s[i+2]))
            else:
                stack.append(int(s[i]))
            i += 1
       
        return stack[-1] 
   