from flask import Flask, render_template, request, redirect, url_for
from sodapy import Socrata
from pandas import *
import numpy as np
import plotly.graph_objects as go
import plotly
from plotly.offline import plot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import bokeh
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
from bokeh.models import HoverTool, ColumnDataSource, Title

app = Flask(__name__)

mytoken = "mlLXY32tr9taoGa3UB9P12kdj"
client = Socrata("data.cms.gov", mytoken, username="lixiangan05572@gmail.com", password="A4HxtN2wUbvcUqR")
r = client.get("tcsp-6e99", limit=20000000)
data = DataFrame(r)
data_drggroup = data.groupby('drg_definition')

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        diagnosis = request.form['diagnosis']
        diagnosis = diagnosis.upper()
        diag_list = []
        for n, g in data_drggroup:
            if diagnosis in n:
                diag_list.append(n)
        if len(diag_list) == 1:
            try:
                div = get_div(diag_list[0])
            except:
                return render_template('index.html')
        elif len(diag_list) > 1 and len(diag_list) < 100:
            diag_list2=[]
            for diag in diag_list:
                diag_list2.append(' '.join(diag.split()[2:]))
            return render_template('select.html',diag_list2=diag_list2)
        else:
            return render_template('index.html', warning = 'Cannot find the diagnosis, \
                                   try to type the medical term (e.g. malignancy instead of cancer)')
        return redirect(url_for('.plot1',div=div,diag=diag_list[0]))

#@app.route('/select',methods=['POST'])
#def select():
#    return render_template('select.html')

@app.route('/plot',methods=['POST'])
def plot():
    diag_sel = request.form.get("diagnosis")
    div_sel = get_div(diag_sel)
    return render_template('plot.html',div=div_sel, diag=diag_sel)
@app.route('/plot1',methods=['GET'])
def plot1():
    div = request.args['div']
    diag = request.args['diag']
    return render_template('plot.html',div=div, diag=diag)

@app.route('/analysis',methods=['POST'])
def analysis():
    state = request.form.get("state_code")
    diag = request.form.get("diagnosis")
    for n_div, g_div in data_drggroup:
        if diag in n_div:
            diag_check = n_div
    data_drgdef_ml = data[data['drg_definition']==diag_check]
    payment = np.asarray([float(payment_str) for payment_str in data_drgdef_ml['average_total_payments']])
    covered = np.asarray([float(covered_str) for covered_str in data_drgdef_ml['average_covered_charges']])
    X = []
    for i,j in zip(payment,covered):
        X.append([i,j])
    X=np.asarray(X)
    scaler = StandardScaler()
    scaler.fit(X)
    X_fit = scaler.transform(X)
    km = KMeans(n_clusters=3)
    km.fit(X_fit)
    data_userdefine=data_drgdef_ml[data_drgdef_ml['provider_state']==state]
    payment_user = np.asarray([float(payment_str) for payment_str in data_userdefine['average_total_payments']])
    covered_user = np.asarray([float(covered_str) for covered_str in data_userdefine['average_covered_charges']])
    hosp_user = data_userdefine['provider_name']
    hosp = data_drgdef_ml['provider_name']
    X_user = []
    for i,j in zip(payment_user,covered_user):
        X_user.append([i,j])
    X_user_f = scaler.transform(X_user) 
    idx = np.argsort(km.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(3)
    TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    colors = []
    d_cluster = {}
    for label in lut[km.labels_]:
        if label == 0:
            colors.append('red')
            if label not in d_cluster:
                d_cluster[label]='red'
        if label == 1:
            colors.append('green')
            if label not in d_cluster:
                d_cluster[label]='green'
        if label == 2:
            colors.append('blue')
            if label not in d_cluster:
                d_cluster[label]='blue'
    source = ColumnDataSource(data=dict(
            x=X_fit[0:,0],
            y=X_fit[0:,1],
            colors=colors,
            a=payment,
            b=covered,
            desc=hosp,
            ))
    source2 = ColumnDataSource(data=dict(
            x2=X_user_f[0:,0],
            y2=X_user_f[0:,1],
            a2 = payment_user,
            b2 = covered_user,
            desc2=hosp_user,
            ))
#    hover = HoverTool(names=['all'],
#                      tooltips = [
#                              ("Ave. payments ($)", "@a{int}"),
#                              ("Ave. covered cost ($)", "@b{int}"),
#                              ("Hospital", "@desc"),
#                              ])
    hover2 = HoverTool(names=['user'],
                       tooltips = [
                               ("Ave. payments ($)", "@a2{int}"),
                               ("Ave. covered cost ($)", "@b2{int}"),
                               ("Hospital", "@desc2"),
                               ])
    p = figure(tools=[TOOLS,hover2],plot_width=800, plot_height=800)
#    p = figure(plot_width=800, plot_height=800)
    p.scatter('x', 'y', source=source, fill_color='colors',line_color=None,name='all')
    p.scatter([0],[0],fill_color=d_cluster[0],line_color=None,size=0,legend_label='Cluster1: Low payments, Low covered cost')
    p.scatter([0],[0],fill_color=d_cluster[1],line_color=None,size=0,legend_label='Cluster2: Low payments, high covered cost')
    p.scatter([0],[0],fill_color=d_cluster[2],line_color=None,size=0,legend_label='Cluster3: High payments')
    p.triangle('x2', 'y2', source=source2,fill_color='black',line_color=None,size=15,legend_label="Hospitals in "+state,name='user')
    p.add_layout(Title(text=diag, text_font_style="italic"), 'above')
    p.add_layout(Title(text='Clustering of hospitals in USA for:', text_font_size="20pt"),  'above')
    p.legend.label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.xaxis.major_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = '20pt'
    p.xaxis.axis_label = 'Standard normalized average payments'
    p.yaxis.axis_label = 'Standard normalized average covered cost'
    p.yaxis.formatter.use_scientific = False
    p.toolbar_location = None
    script, div = components(p)
    return render_template('analysis.html',script=script,div=div)



@app.route('/about')
def about():
    return render_template('about.html')

def get_div(diag_gd):
    for n_div, g_div in data_drggroup:
        if diag_gd in n_div:
            diag_gd_check = n_div
    data_drgdef = data[data['drg_definition']==diag_gd_check]
    data_drgdef = data_drgdef.groupby('provider_state')
    d_drgdef_state_plot = []
    d_drgdef_avecost_plot = []
    for name, group in data_drgdef:
        cost = [float(cost_str) for cost_str in data_drgdef.get_group(name)['average_total_payments'].values]
        case = [float(case_str) for case_str in data_drgdef.get_group(name)['total_discharges'].values]
        cost = np.asarray(cost)
        case = np.asarray(case)
        d_drgdef_avecost_plot.append(np.dot(cost,case)/sum(case))
        d_drgdef_state_plot.append(name)
        
        fig = go.Figure(data=go.Choropleth(
                locations=d_drgdef_state_plot, # Spatial coordinates
                z = d_drgdef_avecost_plot, # Data to be color-coded
                locationmode = 'USA-states', # set of locations match entries in `locations`
                colorscale = 'Reds',
                colorbar_title = "USD",
                ))
        diag_plot = ' '.join(diag_gd_check.split()[2:])
        fig.update_layout(title_text = 'Average total cost for '+\
                          diag_plot,geo_scope='usa',autosize=False, width=1200,height=550,)
        
        config = {'toImageButtonOptions':
            {'width': '60%',
             'height': '60%',
             'format': 'svg',
             'filename': 'fig_diagnosis'}}
        div_gd = plotly.offline.plot(fig, config=config,auto_open=False,\
                                  show_link=False,include_plotlyjs=False, output_type='div')
    return div_gd

if __name__ == '__main__':
 app.run(port=33507)
