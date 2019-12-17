from django import forms
import datetime

from .models import AgencyMaster,AgencyProductMaster,AgencyProductRiskMaster,Agency_Prod_Comm_Master,AgencyVehicleMaster,AgencyVehicleDepriciation,AgencyNcbMaster,AgencyClaimStatusMaster,AgencyClaimsSurveyorMaster


'''class AgencyMasterForm(forms.ModelForm):
    class Meta:
        
        model=AgencyMaster
        fields = '__all__'
        exclude=('created_by','last_updated_by')'''






class ProductMaster(forms.ModelForm):
    prod_start_date=forms.DateField(widget=forms.SelectDateWidget(),initial=datetime.date.today())
    prod_end_date=forms.DateField(widget=forms.SelectDateWidget())
    class Meta:
        model=AgencyProductMaster
        fields=['prod_code',
                'prod_description',
                'prod_start_date',
                'prod_end_date',
                'created_date',
                'last_update_date'

                ]
        exclude=['created_by ','last_updated_by ']


    def clean_prod_code(self,*args,**kwargs):
        data=self.cleaned_data['prod_code']
        if not data.isupper():
            raise forms.ValidationError('Prodcode should in upper case')
        if '@' in data or '^' in data or '$' in data:
            raise forms.ValidationError('prodcode does not contain special letter')
        if len(data)<5:
            raise forms.ValidationError('more than 5 charecter')
        return data
    def clean_prod_description(self,*args,**kwargs):
        data=self.cleaned_data['prod_description']
        if len(data)<10:
            raise forms.ValidationError('more than 10 charecter')
        return data


class Product_Risk_Form(forms.ModelForm):
    cal_method=(('','______'),
                ('F','Fixed'),
                ('P','Percentage'))
    risk_start_date = forms.DateField(widget=forms.SelectDateWidget(),required=True)
    risk_end_date = forms.DateField(widget=forms.SelectDateWidget(),required=True)
    risk_code = forms.CharField(widget=forms.TextInput(attrs={'size':10}))
    risk_description =  forms.CharField(widget=forms.TextInput(attrs={'size':10}))
    risk_premium_percent = forms.DecimalField(required=False)
    prem_calc_method = forms.ChoiceField(choices=cal_method,required=False)

    class Meta:
        model=AgencyProductRiskMaster
        fields=['risk_code',
                'prod_code',
                'risk_description',
                'risk_premium_percent',
                'risk_start_date',
                'risk_end_date',
                'created_date',
                'last_update_date',
                'prem_calc_method',
                'fixed_prem',
                'fixed_si']
        exclude=['created_by','last_updated_by']















