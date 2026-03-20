"""
TriageDelRo ML Backend — Model Training
Models:
  1. Category Classifier  → Flooding | Fire | Medical Emergency | Infrastructure Damage
  2. Severity Classifier  → High | Medium | Low
  3. Credibility Scorer   → credible | suspicious

Run: python train_model.py
"""

import os, pickle, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — CATEGORY + SEVERITY
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_SAMPLES = [
    # FLOODING HIGH
    ("Baha na mataas sa Purok 3 hindi na makalabas ang mga tao", "Flooding", "High"),
    ("Flash flood warning street completely underwater families trapped", "Flooding", "High"),
    ("Families trapped on rooftops due to rising floodwater need rescue", "Flooding", "High"),
    ("Malakas na ulan baha na hanggang dibdib sa Purok 5", "Flooding", "High"),
    ("Flood submerged entire road vehicles cannot pass people stranded", "Flooding", "High"),
    ("Rising waters entering homes 20 families need evacuation now", "Flooding", "High"),
    ("Mataas na baha nakalubog na ang bahay sa Sitio Maligaya", "Flooding", "High"),
    ("Flood level critical barangay road impassable elderly trapped", "Flooding", "High"),
    ("Umaapaw na ang ilog baha papasok na sa mga bahay", "Flooding", "High"),
    ("Severe flooding rescue boats needed at Purok 4 urgent", "Flooding", "High"),
    ("Walang makalabas dahil sa mataas na baha sa gate ng barangay", "Flooding", "High"),
    ("Tumataas ang tubig sa buong barangay baha na lahat", "Flooding", "High"),
    ("Hindi na makita ang kalsada lahat ay tubig na", "Flooding", "High"),
    ("Flood reaching second floor of houses in sitio 2", "Flooding", "High"),
    ("Rapid flooding from river overflow 30 families affected", "Flooding", "High"),
    ("Baha na hanggang baywang hindi makalabas ang mga bata", "Flooding", "High"),
    ("Naanod ang tricycle sa baha mataas na tubig", "Flooding", "High"),
    ("Tubig na pumapasok sa bahay kailangan ng tulong agad", "Flooding", "High"),

    # FLOODING MEDIUM
    ("Nagbabaha sa kalsada mababa pero nakakaistorbo sa trapiko", "Flooding", "Medium"),
    ("Street flooding after heavy rain ankle deep water", "Flooding", "Medium"),
    ("Baha sa kalsada pwede pa lumakad pero mapanganib na", "Flooding", "Medium"),
    ("Flood water rising slowly residents on alert", "Flooding", "Medium"),
    ("Low lying area flooding some residents preparing to evacuate", "Flooding", "Medium"),
    ("Pagbaha sa purok moderate level nakaapekto sa ilang bahay", "Flooding", "Medium"),
    ("Knee high flooding in the market area businesses affected", "Flooding", "Medium"),
    ("Patuloy na umaangat ang tubig sa ilang bahagi ng barangay", "Flooding", "Medium"),
    ("Moderate baha sa Purok 6 nagsisimula nang pumasok sa bahay", "Flooding", "Medium"),
    ("Flooding in barangay hall area moderate depth residents watching", "Flooding", "Medium"),

    # FLOODING LOW
    ("Slight flooding sa kanto pagkatapos ng ulan mabilis mawawala", "Flooding", "Low"),
    ("Minor water accumulation on road draining slowly", "Flooding", "Low"),
    ("Maliit na baha sa pasukan ng barangay hindi malubha", "Flooding", "Low"),
    ("Small puddles forming after rain not dangerous yet", "Flooding", "Low"),
    ("Konting tubig sa kalsada lilipas din pagtirik ng araw", "Flooding", "Low"),
    ("Light flooding near creek minor inconvenience only", "Flooding", "Low"),

    # FIRE HIGH
    ("Nasusunog ang tatlong bahay sa Purok 2 tumakas na ang mga residente", "Fire", "High"),
    ("House fire spreading to adjacent structures multiple families displaced", "Fire", "High"),
    ("Malaking sunog naglalagablab na ang apartment sa may palengke", "Fire", "High"),
    ("Fire engulfing warehouse thick smoke visible from barangay", "Fire", "High"),
    ("Sunog sa Purok 7 walang makalapit dahil sa init at usok", "Fire", "High"),
    ("Major fire 5 houses burning residents evacuating now", "Fire", "High"),
    ("Nasusunog ang LPG store panganib sa mga kapitbahay", "Fire", "High"),
    ("Structure fire with reported casualties fire trucks needed urgent", "Fire", "High"),
    ("Nasabog ang kalan may sunog na sa bahay tumakas na lahat", "Fire", "High"),
    ("Sunog tumakas na kami malaki na ang apoy", "Fire", "High"),
    ("Fire spreading fast to 4 houses wind pushing flames", "Fire", "High"),
    ("Malaking apoy sa bodega malapit sa paaralan", "Fire", "High"),
    ("Nasunog na ang dalawang bahay patuloy pa lumalaki", "Fire", "High"),
    ("Sunog hindi na mapigilan kailangan ng fire truck agad", "Fire", "High"),

    # FIRE MEDIUM
    ("Sunog sa bakuran apoy kontrolado pa pero lumalaki", "Fire", "Medium"),
    ("Small house fire one room affected neighbors helping", "Fire", "Medium"),
    ("Fire in vacant lot spreading slowly toward houses", "Fire", "Medium"),
    ("Nasusunog ang isang bahay napatay na ang isa pero hindi pa ganap", "Fire", "Medium"),
    ("Grass fire near residential area wind pushing it closer", "Fire", "Medium"),
    ("Kitchen fire spreading to living room neighbors assisting", "Fire", "Medium"),
    ("Di pa malaking sunog pero lumalaki kailangan ng tulong", "Fire", "Medium"),

    # FIRE LOW
    ("Maliit na sunog sa basurahan napatay na ng mga residente", "Fire", "Low"),
    ("Minor fire in kitchen extinguished by homeowner", "Fire", "Low"),
    ("Nasunog ang mga dahon sa bakuran kontrolado at maliit lang", "Fire", "Low"),
    ("Small campfire got out of hand neighbors put it out", "Fire", "Low"),
    ("Burning trash near road already controlled by residents", "Fire", "Low"),

    # MEDICAL EMERGENCY HIGH
    ("May naaksidente sa kalsada may sugat at walang malay", "Medical Emergency", "High"),
    ("Person collapsed unconscious on the street not breathing", "Medical Emergency", "High"),
    ("Matanda na nahihirapan huminga kailangan ng ambulansya agad", "Medical Emergency", "High"),
    ("Bata nalunod sa baha di pa nakakagising", "Medical Emergency", "High"),
    ("Multiple casualties in road accident urgent medical attention needed", "Medical Emergency", "High"),
    ("Lalaki tinamaan ng kidlat di na gumagalaw", "Medical Emergency", "High"),
    ("Heart attack suspected elderly patient unresponsive", "Medical Emergency", "High"),
    ("Serious injury from falling debris bleeding heavily", "Medical Emergency", "High"),
    ("Naaksidente ang tricycle driver hindi makakilos dumugo ang ulo", "Medical Emergency", "High"),
    ("Child drowning in flood water unconscious", "Medical Emergency", "High"),
    ("Pregnant woman in labor no transport to hospital", "Medical Emergency", "High"),
    ("Man hit by falling tree pinned cannot move", "Medical Emergency", "High"),
    ("Matanda hindi na humihinga kailangan ng tulong ngayon", "Medical Emergency", "High"),
    ("Aksidente sa kalsada maraming sugatan", "Medical Emergency", "High"),

    # MEDICAL EMERGENCY MEDIUM
    ("May na food poison sa selebrasyon ilang tao may sakit tiyan", "Medical Emergency", "Medium"),
    ("Person injured in minor accident walking but in pain", "Medical Emergency", "Medium"),
    ("Ilang bata may lagnat at ubo posibleng dengue", "Medical Emergency", "Medium"),
    ("Elderly fell down possible broken bone but conscious", "Medical Emergency", "Medium"),
    ("Minor laceration from broken glass needs stitches", "Medical Emergency", "Medium"),
    ("Asthma attack patient conscious but struggling to breathe", "Medical Emergency", "Medium"),
    ("Nahulog mula sa puno may posibleng bali ang braso", "Medical Emergency", "Medium"),
    ("Several residents sick after community meal vomiting", "Medical Emergency", "Medium"),

    # MEDICAL EMERGENCY LOW
    ("Bata may sakit kailangan ng gamot pero hindi emergency", "Medical Emergency", "Low"),
    ("Minor cut from accident needs first aid", "Medical Emergency", "Low"),
    ("Resident needs medical consultation not urgent", "Medical Emergency", "Low"),
    ("Mild fever requesting barangay health center assistance", "Medical Emergency", "Low"),
    ("Napaso ng mainit na pagkain minor burn lang", "Medical Emergency", "Low"),

    # INFRASTRUCTURE DAMAGE HIGH
    ("Bumagsak ang malaking puno sa kalsada naharang ang daan", "Infrastructure Damage", "High"),
    ("Bridge structurally damaged dangerous to cross", "Infrastructure Damage", "High"),
    ("Electrical post knocked down by storm live wires on road", "Infrastructure Damage", "High"),
    ("Malaking butas sa kalsada nahulog na ang motor", "Infrastructure Damage", "High"),
    ("Retaining wall collapsed blocking main road to barangay", "Infrastructure Damage", "High"),
    ("Nabasag ang tubo ng tubig walang tubig ang buong Purok 6", "Infrastructure Damage", "High"),
    ("Landslide blocking road several houses at risk", "Infrastructure Damage", "High"),
    ("Downed power lines sparking on flooded road extreme danger", "Infrastructure Damage", "High"),
    ("Bumagsak ang tulay hindi na pwedeng tawirin", "Infrastructure Damage", "High"),
    ("Collapsed wall blocking evacuation route", "Infrastructure Damage", "High"),
    ("Kuryente bumagsak sa kalsada mapanganib", "Infrastructure Damage", "High"),
    ("Gumuho ang lupa naharang ang daan", "Infrastructure Damage", "High"),

    # INFRASTRUCTURE DAMAGE MEDIUM
    ("Damaged road with large potholes affecting vehicles", "Infrastructure Damage", "Medium"),
    ("Street light broken sa may intersection mapanganib sa gabi", "Infrastructure Damage", "Medium"),
    ("Nabali ang fence ng barangay hall dahil sa bagyo", "Infrastructure Damage", "Medium"),
    ("Footbridge damaged residents taking longer route", "Infrastructure Damage", "Medium"),
    ("Clogged drainage causing minor flooding needs cleaning", "Infrastructure Damage", "Medium"),
    ("Nasirang gate ng covered court kailangang ayusin agad", "Infrastructure Damage", "Medium"),
    ("Road crack getting wider dangerous for motorcycles", "Infrastructure Damage", "Medium"),

    # INFRASTRUCTURE DAMAGE LOW
    ("Maliit na butas sa kalsada kailangan ng patching", "Infrastructure Damage", "Low"),
    ("Minor damage to barangay signage after storm", "Infrastructure Damage", "Low"),
    ("Basura nakaharang sa kanal kailangan i-clear", "Infrastructure Damage", "Low"),
    ("Broken sidewalk tile minor trip hazard", "Infrastructure Damage", "Low"),
    ("Missing road sign near school zone", "Infrastructure Damage", "Low"),
    ("Damaged drainage cover minor safety concern", "Infrastructure Damage", "Low"),

    # EXTRA FLOODING HIGH
    ("Baha na lahat ng daan walang makalusot", "Flooding", "High"),
    ("Ilog umaapaw baha na sa buong sitio", "Flooding", "High"),
    ("Matindi ang baha kailangan na ng boat rescue", "Flooding", "High"),
    ("Naanod na ang ilang sasakyan sa baha", "Flooding", "High"),
    ("Flood waters rising cannot stop need help now", "Flooding", "High"),

    # EXTRA FLOODING MEDIUM
    ("Baha sa kalsada hindi pa malubha pero nakaabala", "Flooding", "Medium"),
    ("Moderate flooding blocking purok road", "Flooding", "Medium"),
    ("Bahang-baha sa ilalim ng tulay", "Flooding", "Medium"),

    # EXTRA FLOODING LOW
    ("Maliit na baha sa gilid ng kalsada", "Flooding", "Low"),
    ("Bahang mababa lang pagkatapos ng ulan", "Flooding", "Low"),

    # EXTRA FIRE HIGH
    ("Sunog patuloy lumalaki kailangan ng fire truck ngayon", "Fire", "High"),
    ("Apoy kumakalat sa mga kapitbahay", "Fire", "High"),
    ("Nasusunog ang mga bahay sa Purok 3", "Fire", "High"),
    ("Malaking sunog sa palengke maraming apektado", "Fire", "High"),

    # EXTRA FIRE MEDIUM
    ("Sunog sa isang kwarto ng bahay", "Fire", "Medium"),
    ("Apoy sa bakuran papalapit sa bahay", "Fire", "Medium"),

    # EXTRA FIRE LOW
    ("Nasunog ang basura sa likod ng bahay kontrolado na", "Fire", "Low"),
    ("Minor fire already put out by residents", "Fire", "Low"),

    # EXTRA MEDICAL HIGH
    ("May nahulog sa ilog hindi na gumagalaw", "Medical Emergency", "High"),
    ("Aksidente sa kalsada may patay posible", "Medical Emergency", "High"),
    ("Matanda hinihingal na kailangan ng doktor agad", "Medical Emergency", "High"),
    ("Cardiac arrest patient needs ambulance now", "Medical Emergency", "High"),

    # EXTRA MEDICAL MEDIUM
    ("Bata may mataas na lagnat kailangan ng gamot", "Medical Emergency", "Medium"),
    ("Resident injured needs medical help not critical", "Medical Emergency", "Medium"),
    ("Ilang residente may sakit pagkatapos ng baha", "Medical Emergency", "Medium"),

    # EXTRA MEDICAL LOW
    ("Kailangan ng medical certificate sa barangay health center", "Medical Emergency", "Low"),
    ("Minor injury from fall needs first aid only", "Medical Emergency", "Low"),

    # EXTRA INFRASTRUCTURE HIGH
    ("Naputol ang linya ng kuryente mapanganib", "Infrastructure Damage", "High"),
    ("Malaking puno bumagsak sa bahay", "Infrastructure Damage", "High"),
    ("Tulay gumuho hindi na ligtas", "Infrastructure Damage", "High"),

    # EXTRA INFRASTRUCTURE MEDIUM
    ("Sira ang ilaw sa kalsada mapanganib sa gabi", "Infrastructure Damage", "Medium"),
    ("Malaking butas sa kalsada peligroso sa sasakyan", "Infrastructure Damage", "Medium"),

    # EXTRA INFRASTRUCTURE LOW
    ("Sira ang kanal kailangan ng ayos", "Infrastructure Damage", "Low"),
    ("Maliit na sira sa kalsada hindi pa urgent", "Infrastructure Damage", "Low"),
]

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — CREDIBILITY
# ══════════════════════════════════════════════════════════════════════════════

CREDIBILITY_SAMPLES = [
    # CREDIBLE
    ("Baha na mataas sa Purok 3 hindi na makalabas ang mga tao", "credible"),
    ("May sunog sa kapitbahay tumakas na kami", "credible"),
    ("Naaksidente ang motor sa kanto may sugatan", "credible"),
    ("Matanda na walang malay sa kalsada kailangan ng tulong", "credible"),
    ("Bumagsak ang puno sa daan hindi makalabas ang sasakyan", "credible"),
    ("Flash flood warning street completely underwater", "credible"),
    ("House fire spreading need fire truck immediately", "credible"),
    ("Person collapsed unconscious not breathing", "credible"),
    ("Electrical post down live wires on street dangerous", "credible"),
    ("Families trapped on rooftop need rescue boats", "credible"),
    ("Nasusunog na ang bahay tumakas na kami tulong", "credible"),
    ("Bridge damaged cannot cross need alternate route", "credible"),
    ("Multiple injuries in road accident near barangay hall", "credible"),
    ("Severe flooding entering homes need evacuation help", "credible"),
    ("LPG tank leaking near fire danger please respond", "credible"),
    ("Bata nalunod sa ilog kailangan ng tulong agad", "credible"),
    ("Road completely blocked by landslide", "credible"),
    ("Heart attack victim unresponsive ambulance needed", "credible"),
    ("Malaking baha hindi na makita ang kalsada", "credible"),
    ("Fire in market area spreading to nearby stalls", "credible"),
    ("Nahulog ang matanda sa hagdan may bali posible", "credible"),
    ("Clogged drainage causing flooding in purok 2", "credible"),
    ("Sunog sa bodega malapit sa mga bahay", "credible"),
    ("Dengue cases rising in barangay health center overwhelmed", "credible"),
    ("Tree fell on house roof family needs help", "credible"),
    ("Tubig na pumasok sa bahay kailangan ng sandbags", "credible"),
    ("Injured resident cannot walk needs transport to hospital", "credible"),
    ("Naaksidente ang bata sa basketball court may dugo", "credible"),
    ("Flood water rising fast near evacuation center", "credible"),
    ("May sunog sa Purok 4 malaking usok", "credible"),
    ("Collapsed fence blocking pedestrian path", "credible"),
    ("Elderly woman needs medical attention at home", "credible"),
    ("Street flooding knee deep near elementary school", "credible"),
    ("Nabasag ang tubo ng tubig bumubuhos sa kalsada", "credible"),
    ("Multiple families displaced by fire need shelter", "credible"),
    ("Child with high fever convulsing needs ambulance", "credible"),
    ("Landslide near sitio blocking evacuation road", "credible"),
    ("Drowning incident at creek near barangay boundary", "credible"),
    ("Pregnant woman in labor need emergency transport", "credible"),
    ("Major road damage after typhoon cannot pass", "credible"),
    ("Baha na lahat ng daan walang makalusot tulong", "credible"),
    ("Matinding sunog tatlong bahay na apektado", "credible"),
    ("Matanda hinihingal na kailangan ng doktor agad", "credible"),
    ("Naputol ang linya ng kuryente mapanganib sa kalsada", "credible"),
    ("Ilog umaapaw baha papasok na sa mga bahay", "credible"),
    ("May nahulog sa ilog hindi na gumagalaw", "credible"),
    ("Aksidente sa kalsada may sugatan marami", "credible"),
    ("Tulay gumuho hindi na ligtas tumawid", "credible"),
    ("Mataas na baha hindi na makita ang mga bahay", "credible"),
    ("Nasunog ang apartment maraming pamilya apektado", "credible"),

    # SUSPICIOUS — PRANK / JOKE
    ("haha sunog daw char", "suspicious"),
    ("baha baha baha lol", "suspicious"),
    ("sunog daw sabi ni juan haha joke lang", "suspicious"),
    ("prank ka char wala namang baha", "suspicious"),
    ("testing lang to wala naman akong irereport", "suspicious"),
    ("hahaha may baha daw sa kanto char char", "suspicious"),
    ("lol baha na daw joke joke", "suspicious"),
    ("sunog si pedro char haha", "suspicious"),
    ("baha ka naman char lang haha", "suspicious"),
    ("may namatay daw haha joke lang bro", "suspicious"),
    ("sunog daw char char char", "suspicious"),
    ("baha na daw lels", "suspicious"),
    ("hehe may aksidente daw char", "suspicious"),
    ("joke lang wala naman talagang sunog", "suspicious"),
    ("prank report wala lang", "suspicious"),

    # SUSPICIOUS — GIBBERISH
    ("asdfghjkl", "suspicious"),
    ("qwerty uiop zxcvbnm", "suspicious"),
    ("123456789", "suspicious"),
    ("aaaaaaaaaaaa", "suspicious"),
    ("xxxxxxxxxxx", "suspicious"),
    ("zzzzzzzzz bbbbbbb", "suspicious"),
    ("hjkl hjkl hjkl", "suspicious"),
    ("1234 test test 5678", "suspicious"),
    ("abcdefghijk lmnop", "suspicious"),
    ("!!!!!!!!!!", "suspicious"),
    ("@@@@@@@@", "suspicious"),
    (".........", "suspicious"),
    ("xxxxxxxx yyyyyyyy", "suspicious"),
    ("zxcvbnm asdfgh", "suspicious"),
    ("00000 11111 22222", "suspicious"),

    # SUSPICIOUS — IRRELEVANT
    ("sino crush mo sa barangay", "suspicious"),
    ("wala lang bored lang ako", "suspicious"),
    ("kumain ka na ba", "suspicious"),
    ("magandang umaga po", "suspicious"),
    ("may pa libreng pagkain ba sa barangay hall", "suspicious"),
    ("kailan ang fiesta", "suspicious"),
    ("sino ang magnanakaw ng tsinelas", "suspicious"),
    ("walang magawa kaya nag rereklamo", "suspicious"),
    ("hello barangay kumusta kayo diyan", "suspicious"),
    ("test report number one two three", "suspicious"),
    ("may laro ba ngayong gabi sa covered court", "suspicious"),
    ("pwede ba mag request ng barangay clearance dito", "suspicious"),
    ("nakita ko si mayor kanina", "suspicious"),
    ("ang init ng panahon ngayon", "suspicious"),
    ("mahal ang bigas ngayon", "suspicious"),

    # SUSPICIOUS — TOO VAGUE
    ("may nangyari", "suspicious"),
    ("grabe", "suspicious"),
    ("ano to", "suspicious"),
    ("ganyan talaga", "suspicious"),
    ("oo nga", "suspicious"),
    ("ok lang", "suspicious"),
    ("wala", "suspicious"),
    ("di ko alam", "suspicious"),
    ("basta may nangyari daw", "suspicious"),
    ("daw daw daw", "suspicious"),
    ("ano ba yan", "suspicious"),
    ("sige na nga", "suspicious"),
    ("ay grabe naman", "suspicious"),
    ("naku po", "suspicious"),
    ("hay nako", "suspicious"),

    # SUSPICIOUS — COMPLAINTS NOT EMERGENCIES
    ("bobo kayo sa barangay hall", "suspicious"),
    ("walang kwenta ang barangay", "suspicious"),
    ("pabaya kayo lagi", "suspicious"),
    ("mag resign na kayo lahat", "suspicious"),
    ("pangit ang trabaho ng BDRRMC", "suspicious"),
    ("wala kayong ginagawa", "suspicious"),
    ("tamad ang mga tanod dito", "suspicious"),
    ("bakit walang streetlight dito matagal na", "suspicious"),
    ("reklamo lang wala kayong nagagawa", "suspicious"),
    ("report ko kayo sa mayor", "suspicious"),
]

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            strip_accents='unicode',
        )),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    os.makedirs('models', exist_ok=True)

    texts = [s[0] for s in CATEGORY_SAMPLES]
    categories = [s[1] for s in CATEGORY_SAMPLES]
    severities = [s[2] for s in CATEGORY_SAMPLES]

    # Category model
    print("Training category model...")
    cat_pipe = build_pipeline()
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        texts, categories, test_size=0.2, random_state=42, stratify=categories
    )
    cat_pipe.fit(Xc_tr, yc_tr)
    print(classification_report(yc_te, cat_pipe.predict(Xc_te)))

    # Severity model
    print("Training severity model...")
    sev_pipe = build_pipeline()
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        texts, severities, test_size=0.2, random_state=42, stratify=severities
    )
    sev_pipe.fit(Xs_tr, ys_tr)
    print(classification_report(ys_te, sev_pipe.predict(Xs_te)))

    # Credibility model
    print("Training credibility model...")
    cred_texts = [s[0] for s in CREDIBILITY_SAMPLES]
    cred_labels = [s[1] for s in CREDIBILITY_SAMPLES]
    cred_pipe = build_pipeline()
    Xt, Xv, yt, yv = train_test_split(
        cred_texts, cred_labels, test_size=0.2, random_state=42, stratify=cred_labels
    )
    cred_pipe.fit(Xt, yt)
    print(classification_report(yv, cred_pipe.predict(Xv)))

    # Save
    with open('models/category_model.pkl', 'wb') as f:
        pickle.dump(cat_pipe, f)
    with open('models/severity_model.pkl', 'wb') as f:
        pickle.dump(sev_pipe, f)
    with open('models/credibility_model.pkl', 'wb') as f:
        pickle.dump(cred_pipe, f)

    with open('models/labels.json', 'w') as f:
        json.dump({
            'categories': ['Flooding', 'Fire', 'Medical Emergency', 'Infrastructure Damage'],
            'severities': ['High', 'Medium', 'Low'],
            'credibility': ['credible', 'suspicious'],
        }, f, indent=2)

    print("\n✅ All 3 models saved to models/")


if __name__ == '__main__':
    main()
