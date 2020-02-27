#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:24:27 2020

@author: hart
"""


from lxml import etree
import smtplib
from email.message import EmailMessage
import gcn
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def time_to_board(time):
    REFTIME = Time(51543.875, format='mjd')
    return (time - REFTIME).sec 

def sent_email(subject, text):
    
    msg = EmailMessage()
    msg['Subject'] = f'{subject}'
    msg['From'] = 'artxc-grb@cosmos.ru'
    msg['To'] = 'i.a.mereminskiy@gmail.com,molkov@iki.rssi.ru'
    msg.set_content(text)

    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


def check_plan(evt_t, evt_ra, evt_dec):
    planfile = open('planned_pointing.dat', 'r')
    data = planfile.readlines()
    t = []
    ra, dec = [],[]
    for line in data:
        if line[0]=='#' or line.strip()=='':
            continue
        L = line.split()
        planned_t = float(L[0])
        if (evt_t- planned_t>30) or (planned_t-evt_t<24*3600):
            t.append(planned_t)
            ra.append(float(L[1]))
            dec.append(float(L[2]))
    coords  = SkyCoord(ra=ra, dec=dec,  frame="fk5", unit="deg")
    evt_pos = SkyCoord(ra=evt_ra, dec=evt_dec,  frame="fk5", unit="deg")
    t      = np.array(t)   
    offset_at_evt = -999            
    if np.min(np.abs(t-evt_t)) < 50.:
        at_evt = np.argmin(np.abs(t-evt_t))
        offset_at_evt = coords[at_evt].separation(evt_pos).degree
    
    min_sep, at_time = -999, -999
    ta  = t[t>evt_t]
    if len(t)>0:
        coords = coords[t>evt_t]
        sepz = coords.separation(evt_pos).degree
        min_sep = np.min(sepz)
        at_time = ta[np.argmin(sepz)] - evt_t
    return offset_at_evt, min_sep, at_time   

def parse_GCN(payload, root):
    notice_role = root.get('role')
    notice_id   = root.get('ivorn')
    payload = payload.decode()    

    content = {}
    for param in root.findall('./What/Param'):
        name = param.attrib['name']
        value = param.attrib['value']
        content[name]=value
    gcn_type = content["Packet_Type"]
    grb_full   = ['61', '98', '99', '53', '54', '55', '100', '101', '102', '111', '112', '115', '131', '121', '127', '128', '134']
    grb_time   = ['110','105', '59', '60', '52', '160', '161']
    grb_all    = grb_full + grb_time
    if (gcn_type in grb_all) and (notice_role=='observation'):
        subject     = f'ART-XC GRB alert from {notice_id}'
        event_time  = root[2].find('.//{*}ISOTime').text
        event_time  = Time(event_time, format='isot')
        obt_time    = time_to_board(event_time)
        ra = -999
        dec = -999
        err = 0
        if gcn_type in grb_full:
            try:
                ra  = float(root[2].find('.//{*}C1').text)
                dec = float(root[2].find('.//{*}C2').text)
                err = float(root[2].find('.//{*}Error2Radius').text)
            except:
                pass
        if ra == -999:
            payload     = f'GRB Notice: burst at {event_time} (approx. {obt_time} OBT), position is unavailable or unrecognised at the moment \n\n\n'+payload    
        else:
            offset_at_evt, min_sep, at_time = check_plan(obt_time, ra, dec)
            buff     = f'GRB Notice: burst at {event_time} (approx. {obt_time} OBT), position ({ra:.4f},{dec:.4f} with error radius {err:.2f} degree)\n'
            pos = SkyCoord(ra=ra, dec=dec, frame="fk5", unit="deg")
            l, b = pos.galactic.l.degree, pos.galactic.b.degree
            half = 'RU'
            if l>180.:
                half = 'DE'
            buff  += f'Galactic (l,b) = ({l:.4f},{b:.4f}), on {half} side\n'
            if offset_at_evt!=-999:
                buff    += f'Angle between tel.axis and burst is {offset_at_evt:.1f}\n'
            if min_sep!=-999:
                buff    += f'In next 24h will pass in {min_sep:.1f} degrees in {(at_time/3600):.1f} hours\n'
            payload  = buff + '\n\n\n'+payload
        sent_email(subject, payload)
    
# Listen for VOEvents until killed with Control-C.
gcn.listen(handler=parse_GCN)
