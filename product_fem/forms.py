import ufl


def dependence_scheme(product_form, coefficient):
    # loop through product_form terms
    in_x_forms, in_y_forms = [], []
    for x_form, y_form in product_form:
        # check if coefficient is in either form
        in_x_forms.append(coefficient in x_form.coefficients())
        in_y_forms.append(coefficient in y_form.coefficients())
    return in_x_forms, in_y_forms

def depends_on(product_form, control):
    is_dependent = []
    for m in control:
        in_x, in_y = dependence_scheme(product_form, m)
        depends = any(in_x) or any(in_y)
        is_dependent.append(depends)
    return any(is_dependent)
    
def derivative(product_form, coefficient, phi):
    """Computes the directional derivative of a ProductForm
    with respect to coefficient in the direction of phi. 
    Returns the derivative as a ProductForm"""
    x, y = [], [] # product_form derivative forms    
    for x_form, y_form in product_form:
        in_x = coefficient in x_form.coefficients()
        in_y = coefficient in y_form.coefficients()
        
        # not worrying about this case yet
        if in_x and in_y:
            raise NotImplementedError
            
        if in_x:
            x.append(ufl.derivative(x_form, coefficient, phi))
            y.append(y_form)
        elif in_y:
            x.append(x_form)
            y.append(ufl.derivative(y_form, coefficient, phi))
        else:
            pass

    return ProductForm(x, y)


class ProductForm:
    
    def __init__(self, x_forms, y_forms):
        if isinstance(x_forms, ufl.form.Form):
            x_forms = [x_forms]
        if isinstance(y_forms, ufl.form.Form):
            y_forms = [y_forms]
        assert len(x_forms)==len(y_forms)
        self.x_forms = x_forms
        self.y_forms = y_forms
#         self.control = control # maybe don't need this
        
    def __len__(self):
        return len(self.x_forms)
    
    def __getitem__(self, item):
        return self.x_forms[item], self.y_forms[item]
    
#     def update(self, control):
#         if self.control is not None:
#             self.control.update(control)
#         else:
#             self.control = control
        
    def function_space(self):
        form = self.x_forms[0]
        return form.arguments()[0].function_space()
    
    
class ProductLinearForm(ProductForm):
    
    def __init__(self, x_forms, y_forms, control=None):
        super().__init__(x_forms, y_forms, control)
        # assert rank 1 forms
        for i in range(len(self)):
            assert len(x_forms[i].arguments())==1
            assert len(y_forms[i].arguments())==1

            
class ProductBilinearForm:
    
    def __init__(self, x_forms, y_forms, control=None):
        super().__init__(x_forms, y_forms, control)
        # assert rank 2 forms
        for i in range(len(self)):
            assert len(x_forms[i].arguments())==2
            assert len(y_forms[i].arguments())==2

            