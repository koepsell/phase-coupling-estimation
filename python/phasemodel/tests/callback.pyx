#
#   cython callback wrapper
#   (adapted from Demos/callback in the cython source code
#

ctypedef void (*cheesefunc)(char *name, void *user_data)

cdef find_cheeses(cheesefunc user_func, void *user_data):
    cdef char* p = 'parmesano'
    user_func(p, user_data)


def find(f):
    find_cheeses(callback, <void*>f)
    
cdef void callback(char *name, void *f):
    (<object>f)(name)
