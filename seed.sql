CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, category TEXT, status TEXT, notes TEXT, created_at DATETIME, updated_at DATETIME);
CREATE TABLE components (id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES projects(id), name TEXT, description TEXT, shop TEXT, url TEXT, estimated_price REAL, quantity INTEGER DEFAULT 1, priority TEXT, status TEXT, created_at DATETIME);
CREATE TABLE shops (id INTEGER PRIMARY KEY, name TEXT, aliases TEXT, category TEXT, notes TEXT);
CREATE TABLE time_blocks (id INTEGER PRIMARY KEY, name TEXT, category TEXT, day_of_week INTEGER, start_time TEXT, duration_min INTEGER, recurring BOOLEAN, notes TEXT);
CREATE TABLE activity_log (id INTEGER PRIMARY KEY, category TEXT, description TEXT, duration_min INTEGER, logged_at DATETIME, project_id INTEGER REFERENCES projects(id));
CREATE TABLE pomodoro_sessions (id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES projects(id), task_description TEXT, started_at DATETIME, ended_at DATETIME, completed BOOLEAN, duration_min INTEGER DEFAULT 25);
-- DATA
INSERT INTO projects (id,name,category,status) VALUES
 (1,'NASGUL','hardware','active'),(2,'Album Lia','music','active'),(3,'Domotica Zigbee','hardware','active'),
 (4,'Classifica auto','software','paused'),(5,'Sito Pentawa','software','done');
INSERT INTO components (project_id,name,shop,estimated_price,quantity,priority,status) VALUES
 (1,'PicoPSU RGEEK 1106','Amazon',22.9,1,'high','to_buy'),
 (1,'HGST Ultrastar 2TB','eBay',35,1,'high','to_buy'),
 (1,'Ventola blower 40mm','AliExpress',4.5,1,'medium','to_buy'),
 (3,'Resistenze 10k','Action',2.0,2,'low','to_buy'),
 (3,'Sonoff Zigbee dongle','Amazon',29.9,1,'high','to_buy'),
 (3,'Pile CR2032','Action',3.5,1,'medium','to_buy'),
 (3,'Sensore Aqara','AliExpress',11,1,'medium','ordered'),
 (1,'Fascette','Action',1.5,1,'low','delivered'),
 (2,'Cavo XLR','Amazon',12,1,'low','cancelled');
INSERT INTO shops (name,aliases,category) VALUES
 ('Action','["action","da action"]','physical'),('Amazon','["amazon","amazon it"]','online'),
 ('AliExpress','["aliexpress","ali"]','online'),('Leroy Merlin','["leroy","leroy merlin"]','physical'),
 ('eBay','["ebay"]','online');
INSERT INTO time_blocks (name,category,day_of_week,start_time,duration_min,recurring) VALUES
 ('Palestra','workout',0,'07:00',60,1),('Palestra','workout',3,'07:00',60,1),('Admin/Banca','admin',5,'10:00',90,1);
INSERT INTO activity_log (category,description,duration_min,logged_at) VALUES
 ('cinema','Visto Arrival',116,'2026-08-02 21:00:00'),('cinema','Visto Dune parte due',166,'2026-09-05 21:00:00'),
 ('gaming','Elden Ring',120,'2026-09-14 22:00:00');
