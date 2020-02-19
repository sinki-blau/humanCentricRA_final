#import all utilities imports
import matplotlib as mp, pandas as pd, numpy as np, geopandas as gpd
import functools
import math
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pysal as ps
import random
import pylab
import matplotlib.colors as cols
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, LineString, MultiLineString
from numpy.random import randn
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.patches as mpatches
import sys
from time import sleep
pd.set_option('precision', 10)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#import all street network fuctions imports
import osmnx as ox, networkx as nx, matplotlib.cm as cm, pandas as pd, numpy as np, geopandas as gpd
import functools
import community
import math
from math import sqrt
import matplotlib.pyplot as plt
import ast

from scipy import sparse
from scipy.sparse import linalg
import pysal as ps

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points
pd.set_option('precision', 10)

#import computational notebook functions
import networkx as nx, matplotlib.cm as cm, pandas as pd, numpy as np
import community
import matplotlib.pyplot as plt
from importlib import reload
import geopandas as gpd
import functools
#%matplotlib inline

pd.set_option('precision', 5)
pd.options.display.float_format = '{:20.2f}'.format
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None

import utilities as uf

##notebook1 

def copy_DiEdges(e):
#u of last row = v of first row
# u = 7, v = 8 
    length = len(e.index)
    #getting column index for items
    col_oneway = e.columns.get_loc('oneway')
    col_u = e.columns.get_loc('u')
    col_v = e.columns.get_loc('v')

    #copying all rows that are not oneway
    for i in range(0,length-1):
        if e.iloc[i, col_oneway] == 0:
            e = e.append(e.iloc[i])

            endRow = -1
            #mapping opposite UV
            e.iloc[endRow ,col_u] = e.iloc[i,col_v]
            e.iloc[endRow ,col_v] = e.iloc[i,col_u]

    # give dataframe new index        
    e.index = np.arange(0,len(e))
    
    return e

def slopeTime(e):
    elen = len(e.index)
    for i in range(0,elen-1):
        grade = e['grade'][i]
        length = e['length'][i]
        if grade > 0:
          rad= grade*100*0.57*(math.pi/180)
          sinrad = math.asin(rad)
          speed = 85*9.81*sinrad+25
          final = (length*speed)/112
          e['slopeCost'][i] = final
        else:
          e['slopeCost'][i] = (length*25)/112
    return e

def weightP1(e):
    elen = len(e.index)
    for i in range(0,elen-1):
        grade = e['grade'][i]
        length = e['length'][i]
        e['weightP1'][i] = (length*25)/112
    return e

#calculating weightP3
def weightP3(e):
    elen = len(e.index)
    for i in range(0,elen-1):
        length = e['length'][i]
        if e['hasBikeP'][i] == 0:
            e['weightP3'][i] = e['slopeCost'][i]+(length*4*0.01)
        else:
            e['weightP3'][i] = e['slopeCost'][i]
    return e 

## Obtaining graph ###############

def DiGraph_fromGDF(nodes_gdf, edges_gdf, nodeID):
    """
    It creates from two geopandas dataframes (street junctions and street segments) a NetworkX graph, passing by a OSMnx function.
    In the lists 'nodes_attributes' and 'edges_costs' please specify attributes that you want to preserve and store in the graph
    representation.
    
    Parameters
    ----------
    nodes_gdf, edges_gdf: GeoDataFrames, nodes and street segments  
    nodes_attributes, edges_costs: lists
    
    Returns
    -------
    GeoDataFrames
    """

    nodes_gdf.set_index(nodeID, drop = False, inplace = True, append = False)
    del nodes_gdf.index.name
    if 'key' in edges_gdf.columns: edges_gdf = edges_gdf[edges_gdf.key == 0].copy()
    
    G = nx.DiGraph()   
    G.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    
    for attribute_name in nodes_gdf.columns:
        if type(nodes_gdf.iloc[0][attribute_name]) == list: 
            attribute_values = {k: v for k, v in attributes[attribute_name].items()}        
        # only add this attribute to nodes which have a non-null value for it
        else: attribute_values = {k: v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_gdf.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        G.add_edge(row['u'], row['v'], **attrs)
    
    return(G)

#notebook 2

#create dual edges from primal gdf
def dual_gdf(nodes_gdf, edges_gdf, crs):
    # computing centroids                                       
    centroids_gdf = edges_gdf.copy()
    centroids_gdf['centroid'] = centroids_gdf['geometry'].centroid
    centroids_gdf['intersecting'] = None

    index_u = centroids_gdf.columns.get_loc("u")+1
    index_v = centroids_gdf.columns.get_loc("v")+1
    index_edgeID = centroids_gdf.columns.get_loc("edgeID")+1

    print('complete 1/6: computing centroids')

    # find_intersecting segments and storing them in the centroids gdf
    processed = []
    for c in centroids_gdf.itertuples():
        intersections = []
        from_node = c[index_u]
        to_node = c[index_v]
        
        #
        possible_intersections = centroids_gdf.loc[(centroids_gdf['u'] == to_node) & (centroids_gdf['v'] != from_node)]

        for p in possible_intersections.itertuples():
            if ((c[0]==p[0]) | ((c[0], p[0]) in processed) | ((p[0], c[0]) in processed)): continue
        
            else:
                intersections.append(p[index_edgeID])  # appending edgeID
                processed.append((p[0],c[0]))
    
        centroids_gdf.set_value(c[0],'intersecting', intersections)
    
    print('complete 2/6: adding intersecting segments')

     # creating vertexes representing street segments (centroids)
    centroids_data = centroids_gdf[['edgeID','streetID', 'intersecting', 'length','grade', 'grade_abs', 'name', 'slopeCost', 'hasBikeP', 'weightP1', 'weightP2', 'weightP3']]
    geometry = centroids_gdf['centroid']
    nodes_dual = gpd.GeoDataFrame(centroids_data, crs=crs, geometry=geometry)
    nodes_dual['x'] = [x.coords.xy[0][0] for x in centroids_gdf['centroid']]
    nodes_dual['y'] = [y.coords.xy[1][0] for y in centroids_gdf['centroid']]

    print('complete 3/6: create centroids as nodes')

    # creating fictious links between centroids
    edges_dual = pd.DataFrame(columns=['u','v', 'geometry'])

    #index_length = nodes_dual.columns.get_loc("length")+1
    index_edgeID_nd = nodes_dual.columns.get_loc("edgeID")+1
    index_intersecting = nodes_dual.columns.get_loc("intersecting")+1
    index_geo = nodes_dual.columns.get_loc("geometry")+1

    print('complete 4/6: create links as edges ')

    # creating vertexes representing street segments (centroids)
    centroids_data = centroids_gdf[['edgeID','streetID', 'intersecting', 'length','grade', 'grade_abs', 'name', 'slopeCost', 'hasBikeP', 'weightP1', 'weightP2', 'weightP3']]
    geometry = centroids_gdf['centroid']
    nodes_dual = gpd.GeoDataFrame(centroids_data, crs=crs, geometry=geometry)
    nodes_dual['x'] = [x.coords.xy[0][0] for x in centroids_gdf['centroid']]
    nodes_dual['y'] = [y.coords.xy[1][0] for y in centroids_gdf['centroid']]

    # creating fictious links between centroids
    edges_dual = pd.DataFrame(columns=['u','v', 'geometry'])

    #part 5 of 6
    index_edgeID_nd = nodes_dual.columns.get_loc("edgeID")+1
    index_intersecting = nodes_dual.columns.get_loc("intersecting")+1
    index_geo = nodes_dual.columns.get_loc("geometry")+1
    # connecting nodes which represent street segments thare a linked in the actual street network                                        
    for row in nodes_dual.itertuples():
        
        edgeID = row[index_edgeID_nd] #streetID of the relative segment
        #length = row[index_length]
                                                    
        # intersecting segments:  # i is the streetID                                      
        for i in list(row[index_intersecting]):     
                  
            # adding a row with u-v, key fixed as 0, Linestring geometry 
            # from the first centroid to the centroid intersecting segment 
            ls = LineString([row[index_geo], nodes_dual.loc[i]['geometry']])
            edges_dual.loc[-1] = [edgeID, i, ls] 
            edges_dual.index = edges_dual.index + 1
            
    edges_dual = edges_dual.sort_index(axis=0)
    geometry = edges_dual['geometry']
    edges_dual = gpd.GeoDataFrame(edges_dual[['u','v']], crs=crs, geometry=geometry)

    print('complete 5/6: connect nodes representing street segments')

    #part 6 of 6 adding angles
    ix_lineA = edges_dual.columns.get_loc("u")+1
    ix_lineB = edges_dual.columns.get_loc("v")+1

    for row in edges_dual.itertuples():

        # retrieveing original lines from/to
        geo_lineA = edges_gdf[edges_gdf.index == row[ix_lineA]].geometry.iloc[0]
        geo_lineB = edges_gdf[edges_gdf.index == row[ix_lineB]].geometry.iloc[0]

        # computing angles in degrees and radians
        deflection = uf.ang_geoline(geo_lineA, geo_lineB, degree = True, deflection = True)
        deflection_rad = uf.ang_geoline(geo_lineA, geo_lineB, degree = False, deflection = True)

        # setting values                                    
        edges_dual.set_value(row[0],'deg', deflection)
        edges_dual.set_value(row[0],'rad', deflection_rad)
    print('complete 6/6: finish adding angles')

    return nodes_dual, edges_dual

# geting dual directed graph
def get_dual_Digraph(nodes_dual, edges_dual):
    """
    The function generates a NetworkX graph from dual-nodes and -edges geopandas dataframes.
            
    Parameters
    ----------
    nodes_dual, edges_dual: GeoDataFrames


    Returns
    -------
    GeoDataFrames
    """
   
    nodes_dual.set_index('edgeID', drop = False, inplace = True, append = False)
    del nodes_dual.index.name
    edges_dual.u = edges_dual.u.astype(int)
    edges_dual.v = edges_dual.v.astype(int)
    
    Dg = nx.DiGraph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
    
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if attribute_name == 'intersecting': continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        Dg.add_edge(row['u'], row['v'], **attrs)

        
    return(Dg)

#get dictionary for dual id
def dual_id_dict(dict_values, graph, nodeAttribute):
    """
    It could be used when one deals with a dual graph and wants to reconnect some analysis conducted on this representation to the
    analysis conducted on the primal graph. For instance, it takes the dictionary containing the betweennes-centrality values of the
    nodes in the dual graph, and associates these features to the corresponding edgeID (nodes in dual graph represent real edges).
    
    Parameters
    ----------
    dict_values: dictionary, of nodeID and centrality values (or other computation)
    G: networkx multigraph
    nodeAttribute: string, attribute of the node to link
    
    Returns
    -------
    dictionary
    """
    
    view = dict_values.items()
    ed_list = list(view)
    ed_dict = {}
    for p in ed_list: ed_dict[graph.node[p[0]][nodeAttribute]] = p[1] #Attribute and measure
        
    return(ed_dict)

#mapping columns in Dual graph 
def mapCol_Dual(n, e, param, name):
    e[name] = 0
    e[name] = e[name].astype('float')
    length = len(e.index)
    for i in range(0,length-1):
        pos = n.edgeID[e.v[i]]
        v_w1 = n.loc[n['edgeID'] == pos, param]
        e[name][i] = v_w1
    return e

#create function to calculate angular confusion
def angConf(e):
    length = len(e.index)
    for i in range(0,length-1):   
        if e['deg'][i] > 2:
            rad = e['rad'][i]
            ang = math.sin(rad/2)**2
            e['angConf'][i] = 10*ang
        else:
            e['angConf'][i] = 0
    return e


#notebook 3

#PRIMAL ROUTING & PLOTTING
#function to map routes back to primal edge
def map_Rto_primalE(rlist,edges):
    #1 from route list create a list with u&v
    edge_nodes = list(zip(rlist[:-1], rlist[1:]))
    r_stID = []

    #2 create for loop which loops the whole edge_nodes
    elen = len(edge_nodes)
    for i in range(0,elen-1):
        row = edges.loc[(edges['u'] == edge_nodes[i][0]) & (edges['v'] == edge_nodes[i][1])]
        strID = row.iloc[0]['edgeID']
        r_stID.append(strID)
    return r_stID

# re-mapping primal route onto dual graph to calculate angular confusion
## input: route_edges, dual edges
def map_PRto_dualE(rEdges,edgesD):
    #1 from route list create a list with u&v
    edge_nodes = list(zip(rEdges['edgeID'].iloc[:-1], rEdges['edgeID'].iloc[1:]))
    edgesPR = edgesD.loc[(edgesD['u'] == edge_nodes[0][0]) & (edgesD['v'] == edge_nodes[0][1])]

    #2 create for loop which loops the whole edge_nodes
    elen = len(edge_nodes)
    for i in range(1,elen-1):
        row = edgesD.loc[(edgesD['u'] == edge_nodes[i][0])  & (edgesD['v'] == edge_nodes[i][1])]
        edgesPR = edgesPR.append(row)
    return edgesPR


# function to create route & gdf that contains edges
#prerequisite: edges has to be mapped with streetID
def map_route(G,edgesP, edgesD, nodes, origin,des,weight):
    route_nodes = nx.shortest_path(G, origin, des, weight)
    edgeIDList = map_Rto_primalE(route_nodes, edgesP)
    routeP_edgesP = edgesP.loc[edgeIDList]
    routeP_edgesD = map_PRto_dualE(routeP_edgesP,edgesD)
    
    #calculate stats
    cycleLen = round((np.sum(routeP_edgesP['length'])),2)
    cycleTime = round((np.sum(routeP_edgesP['weightP1']))/60,2) if weight == 'weightBike' else round((np.sum(routeP_edgesP[weight]))/60,2)
    uphill = round(np.sum(routeP_edgesP[routeP_edgesP['grade']>0]['grade']*routeP_edgesP[routeP_edgesP['grade']>0]['length']),2)
    downhill = round(np.sum(routeP_edgesP[routeP_edgesP['grade']<0]['grade_abs']*routeP_edgesP[routeP_edgesP['grade']<0]['length']),2)
    
    angular_change = round((np.sum(routeP_edgesD['angConf'])),2)

    print('Time to cycle through {}m with {} is {}min. \nCycled uphill for {}m, downhill for {}m, time confused by angular change is {}s. '.format(cycleLen, weight, cycleTime, uphill, downhill, angular_change))
    return routeP_edgesP, routeP_edgesD

# function to create route & gdf that contains edges
#prerequisite: edges has to be mapped with streetID
def map_route2(G,edgesP, edgesD, nodes, origin,des,weight):
    route_nodes = nx.shortest_path(G, origin, des, weight)
    edgeIDList = map_Rto_primalE(route_nodes, edgesP)
    routeP_edgesP = edgesP.loc[edgeIDList]
    routeP_edgesD = map_PRto_dualE(routeP_edgesP,edgesD)
    
    #calculate stats
    cycleLen = round((np.sum(routeP_edgesP['length'])),2)
    cycleTime = round((np.sum(routeP_edgesP['weightP1']))/60,2) if weight == 'weightBike' else round((np.sum(routeP_edgesP[weight]))/60,2)
    uphill = round(np.sum(routeP_edgesP[routeP_edgesP['grade']>0]['grade']*routeP_edgesP[routeP_edgesP['grade']>0]['length']),2)
    downhill = round(np.sum(routeP_edgesP[routeP_edgesP['grade']<0]['grade_abs']*routeP_edgesP[routeP_edgesP['grade']<0]['length']),2)
    
    angular_change = round((np.sum(routeP_edgesD['angConf'])),2)

    print('Time to cycle through {}m with {} is {}min. \nCycled uphill for {}m, downhill for {}m, time confused by angular change is {}s. '.format(cycleLen, weight, cycleTime, uphill, downhill, angular_change))
    return routeP_edgesP, routeP_edgesD, route_nodes

#create bounding box for plotting
def route_bbox(edges):
    margin = 0.05
    west, south, east, north = edges.total_bounds
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    southm = south - margin_ns
    northm = north + margin_ns
    westm = west - margin_ew
    eastm = east + margin_ew
    return southm, northm, westm, eastm  

#plotting routes
def plot_routes(routeN, routeC, route_list, base, title):
    fig, axs = plt.subplots(1,routeN,figsize=(routeN*5, 5))
    fig.suptitle(title, fontdict={'fontsize': '15', 'fontweight' : '3'})
    axs = axs.ravel()

    for i in range(routeN):
        edgesGdf = route_list[i]
        axs[i].axis('off')
        axs[i].set_title('Route '+str(i+1), fontdict={'fontsize': '12', 'fontweight' : '3'})
        southm, northm, westm, eastm = route_bbox(edgesGdf) 
        axs[i].set_ylim((southm, northm))
        axs[i].set_xlim((westm, eastm))

        base.plot(ax=axs[i], color='grey',linewidth=0.5)
        edgesGdf.plot(ax=axs[i], color = routeC ,linewidth=2)
    
    return fig

#dual graph mapping 
#mapping route on dual edges
def map_Rto_dualE(rlist,edges):
    #1 from route list create a list with u&v
    edge_nodes = list(zip(rlist[:-1], rlist[1:]))
    edgesDR = edges.loc[(edges['u'] == edge_nodes[0][0]) & (edges['v'] == edge_nodes[0][1])]

    #2 create for loop which loops the whole edge_nodes
    elen = len(edge_nodes)
    for i in range(1,elen-1):
        row = edges.loc[(edges['u'] == edge_nodes[i][0])  & (edges['v'] == edge_nodes[i][1])]
        edgesDR = edgesDR.append(row)
    return edgesDR

#mapping route & stats
def map_dual_route(G,edgesD, edgesP, nodesP, origin,des,weight):
    route_nodes = nx.shortest_path(G, origin, des, weight)
    edgesDR = map_Rto_dualE(route_nodes,edgesD)
    edgesPR = edgesP.loc[route_nodes]
    
    #calculate stats
    cycleLen = round((np.sum(edgesPR['length'])),2)
    cycleTime = round((np.sum(edgesDR['weightA2']))/60,2) if weight == 'weightA1' else round((np.sum(edgesDR[weight]))/60,2)
    elevation_gained = round((np.sum(edgesPR['grade'])),2)
    uphill = round(np.sum(edgesPR[edgesPR['grade']>0]['grade']*edgesPR[edgesPR['grade']>0]['length']),2)
    downhill = round(np.sum(edgesPR[edgesPR['grade']<0]['grade_abs']*edgesPR[edgesPR['grade']<0]['length']),2)
    
    angular_change = round((np.sum(edgesDR['angConf'])),2)
    print('Time to cycle through {}m with {} is {}min. \nCycled uphill for {}m, downhill for {}m, time confused by angular change is {}s.'.format(cycleLen, weight, cycleTime, uphill, downhill,angular_change))
    return edgesDR, edgesPR

#mapping route & stats
def map_dual_route2(G,edgesD, edgesP, nodesP, origin,des,weight):
    route_nodes = nx.shortest_path(G, origin, des, weight)
    edgesDR = map_Rto_dualE(route_nodes,edgesD)
    edgesPR = edgesP.loc[route_nodes]
    
    #calculate stats
    cycleLen = round((np.sum(edgesPR['length'])),2)
    cycleTime = round((np.sum(edgesDR['weightA2']))/60,2) if weight == 'weightA1' else round((np.sum(edgesDR[weight]))/60,2)
    elevation_gained = round((np.sum(edgesPR['grade'])),2)
    uphill = round(np.sum(edgesPR[edgesPR['grade']>0]['grade']*edgesPR[edgesPR['grade']>0]['length']),2)
    downhill = round(np.sum(edgesPR[edgesPR['grade']<0]['grade_abs']*edgesPR[edgesPR['grade']<0]['length']),2)
    
    angular_change = round((np.sum(edgesDR['angConf'])),2)
    print('Time to cycle through {}m with {} is {}min. \nCycled uphill for {}m, downhill for {}m, time confused by angular change is {}s.'.format(cycleLen, weight, cycleTime, uphill, downhill,angular_change))
    return edgesDR, edgesPR, route_nodes

# visualize by route
def plot_2x2routes(routeN, routeC, route_list, base, title):
    fig, axs = plt.subplots(2,2, figsize=(20,15))
    fig.suptitle(title, fontdict={'fontsize': '15', 'fontweight' : '3'})
    axs = axs.ravel()

    for i in range(routeN):
        edgesGdf = route_list[i]
        axs[i].axis('off')
        axs[i].set_title('Route '+str(i+1), fontdict={'fontsize': '12', 'fontweight' : '3'})
        southm, northm, westm, eastm = route_bbox(edgesGdf) 
        axs[i].set_ylim((southm, northm))
        axs[i].set_xlim((westm, eastm))

        base.plot(ax=axs[i], color='grey',linewidth=0.5)
        edgesGdf.plot(ax=axs[i], color = routeC ,linewidth=2)
    
    return fig

def plot_sigroute(routeC, edgesGdf, base, title):
    fig, ax = plt.subplots(figsize=(10,10))
    #fig.suptitle(title, fontdict={'fontsize': '15', 'fontweight' : '3'})
    #axs = axs.ravel()

    #for i in range(routeN):
    #edgesGdf = route_list[i]
    ax.axis('off')
    ax.set_title(title, fontdict={'fontsize': '12', 'fontweight' : '3'})
    southm, northm, westm, eastm = route_bbox(edgesGdf) 
    ax.set_ylim((southm, northm))
    ax.set_xlim((westm, eastm))

    base.plot(ax=ax, color='grey',linewidth=0.5)
    edgesGdf.plot(ax=ax, color = routeC ,linewidth=2)
    
    return fig

#create parameter table
def addCol_toTable(routedf):
    routedf['length'] = 0
    routedf['time']= 0
    routedf['uphill'] = 0
    routedf['downhill'] = 0
    routedf['angConf'] = 0
    routedf['bikeP'] = 0

    routedf.length = routedf.length.astype(float)
    routedf.time = routedf.time.astype(float)
    routedf.uphill = routedf.uphill.astype(float)
    routedf.downhill = routedf.downhill.astype(float)
    routedf.bikeP = routedf.bikeP.astype(float)
    routedf.angConf = routedf.angConf.astype(float)
    
    return routedf

#prerequesite: route, edge, weight array
# add stats to table
def stats_toTable(routedf, rArray, eArray):
    #adding length, uphill, downhill, bikeP%
    for i in range(len(rArray)):
        routedf.length[i] = round((np.sum(rArray[i]['length'])),2)
        routedf.uphill[i] = round(np.sum(rArray[i][rArray[i]['grade']>0]['grade']*rArray[i][rArray[i]['grade']>0]['length']),2)
        routedf.downhill[i] = round(np.sum(rArray[i][rArray[i]['grade']<0]['grade_abs']*rArray[i][rArray[i]['grade']<0]['length']),2)
        routedf.bikeP[i] = round((np.sum(rArray[i].loc[rArray[i]['hasBikeP']==1]['length'])/routedf.length[i])*100,2)

    #adding angular confusion in seconds
    for i in range(len(eArray)):
        routedf.angConf[i] = round((np.sum(eArray[i]['angConf'])),2)

    #adding cycling time 
    #for primal routes need to calculated by primal edge, dual on dual edges
    for i in range(0,4):
        routedf.time[i] = round((np.sum(rArray[i]['weightP1']))/60,2) if weight[i] == 'weightBike' else round((np.sum(rArray[i][str(weight[i])]))/60,2)
    for i in range(4, len(edges1s)):
        routedf.time[i] = round((np.sum(eArray[i]['weightA2']))/60,2) if weight[i] == 'weightA1' else round((np.sum(eArray[i][str(weight[i])]))/60,2)    
    
    return routedf

#save routes
def save_route(folderName, rArray, nameArray):
    s_path = 'dataComp/'+ folderName+'/'
    crs = {'init': 'epsg:4326', 'no_defs': True}
    
    for i in range(len(rArray)):
        rArray[i].crs = crs
        rArray[i].to_file(s_path+nameArray[i]+'.shp', driver='ESRI Shapefile')