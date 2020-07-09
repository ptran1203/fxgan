import requests 

users = \
"""Catherine Padaca	SPRCATP1	AV
Monique Remetilla	SPRMONR1	V
Johna Comaya	SPRJOHC3	V
Jane Mangantulao	SPRJANM3	V
Joy Sadagnot	SPRJOYS2	V
Rose Penilla	SPRROSP1	V
Regie Balmores	SPRREGB1	AV
Bong Gutierrez	SPRBONG1	A
Adrian Agawin	SPRADRA2	V
Alexander Delossantos	SPRALED2	V
Axelrigo Aliponga	SPRAXEA1	V
Danica Patanao	SPRDANP1	V
Dicky Tampus	SPRDICT1	V
Earl Taboada	SPREART1	V
Gerald Corpuz	SPRGERC2	V
Glicemae Montecillo	SPRGLIM1	V
Jasper Solon	SPRJASS2	V
Jayron Sualog	SPRJAYS4	V
Jayson Deguzman	SPRJAYD1	V
Ladyshane Patlingarao	SPRLADP1	V
Marjorie Villasis	SPRMARV3	V
Markjason Adaza	SPRMARA14	V
Melody Abonal	SPRMELA4	V
Meril Delavin	SPRMERD2	V
Mikegerard Coyoca	SPRMIKC1	V
Nina Benabente	SPRNINB1	V
Razelann Palmones	SPRRAZP1	V
Renan Baring	SPRRENB1	V
Richard Cabugawan	SPRRICC1	V
Rodel Faustino	SPRRODF1	V
Rosalinda Berdin	SPRROSB2	V
Ruvill Irinco	SPRRUVI1	V
Ryan Miranda	SPRRYAM1	V
Zarbie Arabit	SPRZARA1	V
Rickcelle Ann Cantuba	SPRRICC4	V
Rudolph De Asis	SPRRUDD1	V
Johnson Samortin	SPRJOHS1	V
Gia Marie Tibubos	SPRGIAT1	V
Irene Osorio	SPRIREO1	V
Mark Keiven Caro	SPRMARC4	V
Ryan Barlizo	SPRRYAB1	AV
Mark Anthony Santos	SPRMARS3	AV
Rhyzel Cortez	SPRRHYC1	AV
Benedict Domagtoy	SPRBEND1	A
Marifi Magboo	SPRMARM2	A
Jennica Paran	SPRJENP1	A"""
role_map = {
    "A": 3,
    "V": 2,
    "AV": 1,
}

def insert(fname, uname, role):
    
for u in users.split("\n"):
    fname, uname, role = u.split("\t")
    role = role_map[role]
    print("=======\nfull name '{}'\nuser name '{}'\nrole '{}'\n".format(
        fname, uname, role
    ))

