def str2num(s):
   ''' Convert the string s to number. '''

   try:
      return int(s)
   except ValueError:
      return float(s)

def str2bool(s):
   ''' Convert the string s to boolean. '''

   if s == 'True':
      return True
   elif s =='False':
      return False
   else:
      raise ValueError
