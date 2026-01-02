import os
import re
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class BusBotV13:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not password:
            print("⚠️ CẢNH BÁO: Chưa cấu hình NEO4J trong file .env")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # --- TỪ ĐIỂN VIẾT TẮT ---
        self.ABBREVIATIONS = {
            "tdt": "đại học tôn đức thắng", "đh tdt": "đại học tôn đức thắng",
            "văn lang": "trường đại học văn lang", "đh văn lang": "trường đại học văn lang", "vlu": "trường đại học văn lang",
            "csnd": "đại học cảnh sát nhân dân", "đh csnd": "đại học cảnh sát nhân dân",
            "bk": "đại học bách khoa", "bách khoa": "đại học bách khoa",
            "khtn": "đại học khoa học tự nhiên", "tự nhiên": "đại học khoa học tự nhiên",
            "ussh": "đại học khoa học xã hội và nhân văn", "nhân văn": "đại học khoa học xã hội và nhân văn",
            "spkt": "đại học sư phạm kỹ thuật", "sư phạm kỹ thuật": "đại học sư phạm kỹ thuật",
            "nlu": "đại học nông lâm", "nông lâm": "đại học nông lâm",
            "ueh": "đại học kinh tế", "kinh tế": "đại học kinh tế",
            "ulu": "đại học luật", "luật": "đại học luật",
            "yds": "đại học y dược", "y dược": "đại học y dược",
            "hutech": "đại học công nghệ tphcm", "công nghệ": "đại học công nghệ tphcm",
            "uef": "đại học kinh tế tài chính",
            "rmit": "đại học rmit", "fpt": "đại học fpt",
            "hoa sen": "đại học hoa sen", "hsu": "đại học hoa sen",
            "sgu": "đại học sài gòn", "sài gòn": "đại học sài gòn",
            "iuh": "đại học công nghiệp", "công nghiệp": "đại học công nghiệp",
            "ntt": "đại học nguyễn tất thành", "nguyễn tất thành": "đại học nguyễn tất thành",
            "ou": "đại học mở", "mở": "đại học mở",
            
            "bến thành": "bến xe buýt sài gòn", "chợ bến thành": "bến xe buýt sài gòn",
            "suối tiên": "khu du lịch suối tiên", "kdl suối tiên": "khu du lịch suối tiên",
            "đầm sen": "công viên văn hóa đầm sen", "cv đầm sen": "công viên văn hóa đầm sen", "công viên nước đầm sen": "công viên văn hóa đầm sen",
            "thảo cầm viên": "thảo cầm viên",
            "nhà thờ đức bà": "công xã paris",
            "sân bay": "sân bay tân sơn nhất", "tsn": "sân bay tân sơn nhất","sân bay tsn": "sân bay tân sơn nhất",
            "chợ rẫy": "bệnh viện chợ rẫy", "115": "bệnh viện nhân dân 115",
            "lotte mart nam sài gòn":"Lotte Mart","lotte mart quận 7":"Lotte Mart",
            
            "đh": "đại học", "trường đh": "đại học",
            "cđ": "cao đẳng", "bv": "bệnh viện",
            "kdc": "khu dân cư", "bx": "bến xe", "q.": "quận ",
        }
        
        sorted_keys = sorted(self.ABBREVIATIONS.keys(), key=len, reverse=True)
        self.pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')

    def close(self): self.driver.close()

    def normalize_query(self, text):
        text = text.lower()
        def replace(match): return self.ABBREVIATIONS[match.group(0)]
        text = self.pattern.sub(replace, text)
        text = text.replace("đại học đại học", "đại học").replace("trường đại học trường đại học", "trường đại học").replace("bến xe bến xe", "bến xe").replace("bệnh viện bệnh viện", "bệnh viện").replace("công viên văn hóa công viên văn hóa", "công viên văn hóa")
        return text

    def get_coordinates(self, place_name):
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {'q': f"{place_name}, Ho Chi Minh City, Vietnam", 'format': 'json', 'limit': 1}
            headers = {'User-Agent': 'BusMapBot/1.0'}
            resp = requests.get(url, params=params, headers=headers, timeout=2)
            data = resp.json()
            if data: return float(data[0]['lat']), float(data[0]['lon']), data[0]['display_name']
        except: pass
        return None, None, None

    def find_nearest_station_by_coords(self, session, lat, lng):
        query = "MATCH (n:BusStop) WITH n, point.distance(point({latitude: n.lat, longitude: n.lng}), point({latitude: $lat, longitude: $lng})) AS dist WHERE dist < 1000 RETURN n ORDER BY dist ASC LIMIT 1"
        result = session.run(query, lat=lat, lng=lng).single()
        return result['n'] if result else None

    def _internal_search(self, session, txt):
        q_exact = "MATCH (n:BusStop) WHERE toLower(n.name) CONTAINS $txt OR toLower(n.search) CONTAINS $txt OR toLower(n.code) = $txt RETURN n LIMIT 5"
        res = list(session.run(q_exact, txt=txt))
        candidates = [r['n'] for r in res]
        
        if not candidates and "đại học" in txt:
            short_txt = txt.replace("đại học", "đh")
            res = list(session.run(q_exact, txt=short_txt))
            candidates = [r['n'] for r in res]

        if not candidates:
            lat, lng, addr = self.get_coordinates(txt)
            if lat and lng:
                nearest = self.find_nearest_station_by_coords(session, lat, lng)
                if nearest: candidates.append(nearest)
        return candidates

    def find_stop_candidates(self, session, query_text):
        clean_query = self.normalize_query(query_text).replace("trạm", "").strip()
        candidates = self._internal_search(session, clean_query)
        if not candidates and clean_query != query_text.strip():
            raw_query = query_text.replace("trạm", "").strip()
            candidates = self._internal_search(session, raw_query)
        return candidates 

    # --- HÀM TÌM ĐƯỜNG (CÓ GIÁ VÉ & TỌA ĐỘ) ---
    def solve_route(self, start_text, end_text):
        with self.driver.session() as session:
            s_candidates = self.find_stop_candidates(session, start_text)
            e_candidates = self.find_stop_candidates(session, end_text)

            if not s_candidates: return {"status": "error", "message": f"❌ Không tìm thấy điểm đi: '{start_text}'"}
            if not e_candidates: return {"status": "error", "message": f"❌ Không tìm thấy điểm đến: '{end_text}'"}

            s_ids = [s['id'] for s in s_candidates]
            e_ids = [e['id'] for e in e_candidates]
            
            # ƯU TIÊN 1: ĐI THẲNG
            q_direct = """
            MATCH (s:BusStop)-[:ON_ROUTE]->(route:BusRoute)<-[:ON_ROUTE]-(e:BusStop)
            WHERE s.id IN $s_ids AND e.id IN $e_ids
            RETURN route.route_no, route.name, s.name, e.name, s.lat, s.lng, e.lat, e.lng,
                   toInteger(COALESCE(route.fares, 7000)) as price
            LIMIT 1
            """
            direct = session.run(q_direct, s_ids=s_ids, e_ids=e_ids).single()
            if direct:
                price = direct['price']
                return {
                    "status": "success",
                    "text": (f"🎯 **TÌM THẤY XE ĐI THẲNG!**\n"
                             f"- Đi xe **{direct['route.route_no']}**: {direct['route.name']}\n"
                             f"- Đón tại: {direct['s.name']} -> Xuống tại: {direct['e.name']}\n"
                             f"- 🎫 **Giá vé:** {price:,}đ"),
                    # Trả về tọa độ để Server tạo Link Google Map
                    "path_coords": [[direct['s.lng'], direct['s.lat']], [direct['e.lng'], direct['e.lat']]]
                }

            # ƯU TIÊN 2: 1 LẦN ĐỔI XE
            q_1_transfer = """
            MATCH (s:BusStop)-[:ON_ROUTE]->(r1:BusRoute)<-[:ON_ROUTE]-(mid:BusStop)
            MATCH (mid)-[:ON_ROUTE]->(r2:BusRoute)<-[:ON_ROUTE]-(e:BusStop)
            WHERE s.id IN $s_ids AND e.id IN $e_ids AND r1 <> r2
            RETURN r1.route_no AS bus1, r2.route_no AS bus2, mid.name, s.name, 
                   s.lat, s.lng, mid.lat, mid.lng, e.lat, e.lng,
                   toInteger(COALESCE(r1.fares, 7000)) as p1,
                   toInteger(COALESCE(r2.fares, 7000)) as p2
            LIMIT 1
            """
            one_stop = session.run(q_1_transfer, s_ids=s_ids, e_ids=e_ids).single()
            if one_stop:
                p1 = one_stop['p1']
                p2 = one_stop['p2']
                total = p1 + p2
                return {
                    "status": "success",
                    "text": (f"🔄 **LỘ TRÌNH 1 LẦN ĐỔI XE**\n"
                             f"1. Tại '{one_stop['s.name']}', đón xe **{one_stop['bus1']}**.\n"
                             f"2. Đến trạm '**{one_stop['mid.name']}**' thì xuống.\n"
                             f"3. Đón tiếp xe **{one_stop['bus2']}** để đi tiếp.\n"
                             f"--------------------\n"
                             f"💰 **Chi phí:**\n"
                             f"- Xe {one_stop['bus1']}: {p1:,}đ\n"
                             f"- Xe {one_stop['bus2']}: {p2:,}đ\n"
                             f"👉 **Tổng cộng:** {total:,}đ"),
                    "path_coords": [
                        [one_stop['s.lng'], one_stop['s.lat']],
                        [one_stop['mid.lng'], one_stop['mid.lat']],
                        [one_stop['e.lng'], one_stop['e.lat']]
                    ]
                }

            # ƯU TIÊN 3: 2 LẦN ĐỔI XE
            q_2_transfer = """
            MATCH (s:BusStop)-[:ON_ROUTE]->(r1:BusRoute)<-[:ON_ROUTE]-(m1:BusStop)
            MATCH (m1)-[:ON_ROUTE]->(r2:BusRoute)<-[:ON_ROUTE]-(m2:BusStop)
            MATCH (m2)-[:ON_ROUTE]->(r3:BusRoute)<-[:ON_ROUTE]-(e:BusStop)
            WHERE s.id IN $s_ids AND e.id IN $e_ids 
              AND r1 <> r2 AND r2 <> r3
            RETURN r1.route_no, m1.name, r2.route_no, m2.name, r3.route_no,
                   s.lat, s.lng, m1.lat, m1.lng, m2.lat, m2.lng, e.lat, e.lng,
                   toInteger(COALESCE(r1.fares, 7000)) as p1,
                   toInteger(COALESCE(r2.fares, 7000)) as p2,
                   toInteger(COALESCE(r3.fares, 7000)) as p3
            LIMIT 1
            """
            two_stops = session.run(q_2_transfer, s_ids=s_ids, e_ids=e_ids).single()
            if two_stops:
                p1, p2, p3 = two_stops['p1'], two_stops['p2'], two_stops['p3']
                total = p1 + p2 + p3
                return {
                    "status": "success",
                    "text": (f"🔀 **LỘ TRÌNH 2 LẦN ĐỔI XE**\n"
                             f"1. Xe **{two_stops['r1.route_no']}** -> Trạm '{two_stops['m1.name']}'.\n"
                             f"2. Đổi xe **{two_stops['r2.route_no']}** -> Trạm '{two_stops['m2.name']}'.\n"
                             f"3. Đổi xe **{two_stops['r3.route_no']}** về đích.\n"
                             f"--------------------\n"
                             f"💰 **Chi phí:**\n"
                             f"- Chặng 1: {p1:,}đ\n"
                             f"- Chặng 2: {p2:,}đ\n"
                             f"- Chặng 3: {p3:,}đ\n"
                             f"👉 **Tổng cộng:** {total:,}đ"),
                    "path_coords": [
                        [two_stops['s.lng'], two_stops['s.lat']],
                        [two_stops['m1.lng'], two_stops['m1.lat']],
                        [two_stops['m2.lng'], two_stops['m2.lat']],
                        [two_stops['e.lng'], two_stops['e.lat']]
                    ]
                }

            return {"status": "error", "message": "❌ Quá xa hoặc không có đường đi xe buýt phù hợp."}