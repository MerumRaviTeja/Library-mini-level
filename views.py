
from rest_framework.viewsets import ModelViewSet
from .serializer import Serializersclass
from .models import *
from django.shortcuts import render
from django.shortcuts import redirect
from bson import json_util
from django.core.serializers.json import DjangoJSONEncoder
import json
import csv
from  django.http import HttpResponse
from django.db import connections

import datetime
class serializer(ModelViewSet):
    serializer_class = Serializersclass
    queryset = Hyderabad.objects.all()

def home(request):
       return render(request,'index.html')

def home1(request):
    saving=dummy.objects.create(id_employee="12345",time=datetime.datetime.now())
    request.session["temporary_id"] = saving.id
    return redirect(home2)

def home2(request):
    return render(request,"index1.html")

def home3(request):
    saving = dummy.objects.filter(id=request.session["temporary_id"])
    return render(request, 'index.html',{'a':saving})
def home4(request):
    dummy.objects.filter(id=request.session["temporary_id"]).update(time1=datetime.datetime.now())
    saving = dummy.objects.get(id=request.session["temporary_id"])
    return render(request, 'index2.html', {'a': saving})

def storing_data(request):
    import os
    csv_path = os.path.join(os.path.dirname(__file__), 'student_marks.csv')

    print(csv_path)
    f = open(csv_path, 'r')
    #'S\practice\practiceproject\practiceapp\student_marks.csv
    for line in f:
            column=line.split(',')
            column[-1]=int(column[-1].rstrip('\n'))

            store = ExcelData.objects.create(name=column[0],subject=column[1],mark=column[2])
    f.close()
    return HttpResponse("Done")

def csv_list_report(request):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="reports.csv"'
        writer = csv.writer(response)
        conn=connections["default"]
        cursor = conn.cursor()
        cursor.execute("select * from auth_user;")
        columns = [column[0] for column in cursor.description]
        writer.writerow(columns)
        n = 100
        sqlData = cursor.fetchmany(n)
        while sqlData:
            for row in sqlData:
                writer.writerow(row)
            sqlData = cursor.fetchmany(n)
        cursor.close()
        return response

def json_file(request):
    conn = connections["default"]
    cursor = conn.cursor()
    res=cursor.execute("select * from auth_user;")
    data = []
    columns = [column[0] for column in cursor.description]
    for row in res:
        print(list(row))
        data.append(
            dict(zip(columns, list(row)))
        )
    json_data = json.dumps(data,cls=DjangoJSONEncoder,ensure_ascii=False, indent=4,default=json_util.default)
    response = HttpResponse(json_data, content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename="reports.json"'
    return response









