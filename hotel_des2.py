# hotel_des2.py
# Discrete-Event Simulation for Hotel Front Desk & Housekeeping
# Author: M365 Copilot for Rosamund Qi Fang Soh
# Standalone, no external deps; used by Streamlit app for visualization.

import math, random, heapq
from collections import deque

class HotelDES2:
    def __init__(self,
                 n_rooms=200,
                 sim_days=7,
                 warmup_days=7,
                 mean_daily_arrivals=80,
                 avg_los_nights=2.0,
                 checkin_hour=15.0,
                 checkout_hour=12.0,
                 fd_service_tri_mins=(3, 6, 10),
                 hk_mean_clean_mins=35,
                 hk_lognorm_sigma=0.5,
                 hk_shift_start=9.0,
                 hk_shift_end=17.0,
                 fd_schedule=None,
                 hk_cleaners_schedule=None,
                 random_seed=42):
        self.n_rooms = n_rooms
        self.sim_days = sim_days
        self.warmup_days = warmup_days
        self.total_days = warmup_days + sim_days
        self.T_end = 24 * self.total_days
        self.mean_daily_arrivals = mean_daily_arrivals
        self.avg_los_nights = avg_los_nights
        self.checkin_hour = checkin_hour
        self.checkout_hour = checkout_hour
        self.fd_service_tri_mins = fd_service_tri_mins
        self.hk_mean_clean_mins = hk_mean_clean_mins
        self.hk_lognorm_sigma = hk_lognorm_sigma
        self.hk_shift_start = hk_shift_start
        self.hk_shift_end = hk_shift_end
        self.random_seed = random_seed

        self._rng = random.Random(self.random_seed)

        if fd_schedule is None:
            def _fd_agents(t):
                hod = (t % 24)
                if 0 <= hod < 8:
                    return 2
                elif 8 <= hod < 12:
                    return 3
                elif 12 <= hod < 20:
                    return 6
                else:
                    return 3
            self.fd_agents = _fd_agents
        else:
            self.fd_agents = fd_schedule

        if hk_cleaners_schedule is None:
            def _hk_cleaners(t):
                hod = (t % 24)
                if self.hk_shift_start <= hod < self.hk_shift_end:
                    return 12
                else:
                    return 0
            self.hk_cleaners = _hk_cleaners
        else:
            self.hk_cleaners = hk_cleaners_schedule

        # State
        self.time = 0.0
        self.event_q = []
        self._eid = 0

        self.front_queue = deque()
        self.front_busy = 0
        self.waiting_for_room = deque()

        self.rooms_VC = set(range(self.n_rooms))
        self.rooms_VD = deque()
        self.rooms_O = dict()

        self.cleaners_busy = 0

        self.guest_counter = 0
        self.guest = dict()

        self.metrics = {
            'fd_wait_times': [],
            'room_wait_times': [],
            'total_to_room_times': [],
            'early_checkins': 0,
            'eligible_early': 0,
            'fd_busy_time': 0.0,
            'hk_busy_time': 0.0,
            'cleaning_queue_obs': [],
            'fd_queue_obs': [],
            'occ_obs': [],
        }
        sigma = self.hk_lognorm_sigma
        # Convert mean & sigma to lognormal mu for Python's lognormvariate
        self.hk_lognorm_mu = math.log(self.hk_mean_clean_mins) - 0.5 * sigma * sigma

    # --- Random samplers using instance RNG ---
    def sample_fd_service_hours(self):
        a, m, b = self.fd_service_tri_mins
        return self._rng.triangular(a, b, m) / 60.0

    def sample_cleaning_hours(self):
        return self._rng.lognormvariate(self.hk_lognorm_mu, self.hk_lognorm_sigma) / 60.0

    def sample_los_nights(self):
        # Exponential (mean=avg_los) then ceil to integer >=1
        val = self._rng.expovariate(1.0 / self.avg_los_nights)
        return max(1, int(math.ceil(val)))

    def poisson(self, lam):
        # Knuth's algorithm with local RNG
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    def schedule(self, t, etype, payload=None):
        if t > self.T_end + 5*24:
            return
        self._eid += 1
        heapq.heappush(self.event_q, (t, self._eid, etype, payload))

    def init_arrivals(self):
        # Hour-of-day arrival weights
        hod_w = [0.01]*24
        for h in range(24):
            if 7 <= h < 11:
                hod_w[h] = 0.03
            elif 11 <= h < 14:
                hod_w[h] = 0.05
            elif 14 <= h < 18:
                hod_w[h] = 0.12
            elif 18 <= h < 22:
                hod_w[h] = 0.06
            else:
                hod_w[h] = 0.02
        s = sum(hod_w)
        hod_w = [w/s for w in hod_w]
        for d in range(self.total_days):
            for h in range(24):
                lam_h = self.mean_daily_arrivals * hod_w[h]
                count = self.poisson(lam_h)
                for _ in range(count):
                    t = d*24 + h + self._rng.random()
                    self.schedule(t, 'arrival', None)

    def within_measure(self, t):
        return t >= self.warmup_days * 24

    def record_time_integrals(self, t_next):
        dt = t_next - self.time
        if dt <= 0:
            return
        self.metrics['fd_busy_time'] += self.front_busy * dt
        self.metrics['hk_busy_time'] += self.cleaners_busy * dt
        if self.within_measure(self.time):
            self.metrics['cleaning_queue_obs'].append((self.time, len(self.rooms_VD)))
            self.metrics['fd_queue_obs'].append((self.time, len(self.front_queue)))
            self.metrics['occ_obs'].append((self.time, len(self.rooms_O)))

    def maybe_start_fd(self):
        while self.front_busy < self.fd_agents(self.time) and self.front_queue:
            gid = self.front_queue.popleft()
            g = self.guest[gid]
            g['fd_start'] = self.time
            svc = self.sample_fd_service_hours()
            g['fd_svc_dur'] = svc
            self.front_busy += 1
            self.schedule(self.time + svc, 'fd_done', gid)

    def assign_room_if_available(self, gid):
        if self.rooms_VC:
            room_id = self.rooms_VC.pop()
        else:
            return False
        self.rooms_O[room_id] = gid
        g = self.guest[gid]
        g['room_id'] = room_id
        g['checkin_time'] = self.time
        nights = g['los_nights']
        checkin_day = int(self.time // 24)
        checkout_day = checkin_day + nights
        checkout_t = checkout_day*24 + self.checkout_hour
        self.schedule(checkout_t, 'checkout', room_id)
        if self.within_measure(self.time):
            fd_wait = g['fd_start'] - g['arrival']
            self.metrics['fd_wait_times'].append(fd_wait)
            room_wait = g['checkin_time'] - g['fd_end']
            self.metrics['room_wait_times'].append(room_wait)
            total_wait = g['checkin_time'] - g['arrival']
            self.metrics['total_to_room_times'].append(total_wait)
            if g['fd_end'] % 24 < self.checkin_hour:
                self.metrics['eligible_early'] += 1
                if g['checkin_time'] % 24 < self.checkin_hour:
                    self.metrics['early_checkins'] += 1
        return True

    def maybe_start_hk(self):
        while self.cleaners_busy < self.hk_cleaners(self.time) and self.rooms_VD:
            room_id = self.rooms_VD.popleft()
            dur = self.sample_cleaning_hours()
            self.cleaners_busy += 1
            self.schedule(self.time + dur, 'clean_done', room_id)

    def handle_arrival(self):
        gid = self.guest_counter
        self.guest_counter += 1
        self.guest[gid] = {
            'arrival': self.time,
            'fd_start': None,
            'fd_end': None,
            'fd_svc_dur': None,
            'los_nights': self.sample_los_nights(),
            'room_id': None,
            'checkin_time': None,
        }
        self.front_queue.append(gid)
        self.maybe_start_fd()

    def handle_fd_done(self, gid):
        self.front_busy -= 1
        g = self.guest[gid]
        g['fd_end'] = self.time
        if not self.assign_room_if_available(gid):
            self.waiting_for_room.append(gid)
        self.maybe_start_fd()

    def handle_checkout(self, room_id):
        if room_id in self.rooms_O:
            del self.rooms_O[room_id]
        self.rooms_VD.append(room_id)
        self.maybe_start_hk()

    def handle_clean_done(self, room_id):
        self.cleaners_busy -= 1
        if self.waiting_for_room:
            gid = self.waiting_for_room.popleft()
            # Assign the just-cleaned room directly to the waiting guest
            self.rooms_O[room_id] = gid
            g = self.guest[gid]
            g['room_id'] = room_id
            g['checkin_time'] = self.time
            nights = g['los_nights']
            checkin_day = int(self.time // 24)
            checkout_day = checkin_day + nights
            checkout_t = checkout_day*24 + self.checkout_hour
            self.schedule(checkout_t, 'checkout', room_id)
            if self.within_measure(self.time):
                fd_wait = g['fd_start'] - g['arrival']
                self.metrics['fd_wait_times'].append(fd_wait)
                room_wait = g['checkin_time'] - g['fd_end']
                self.metrics['room_wait_times'].append(room_wait)
                total_wait = g['checkin_time'] - g['arrival']
                self.metrics['total_to_room_times'].append(total_wait)
                if g['fd_end'] % 24 < self.checkin_hour:
                    self.metrics['eligible_early'] += 1
                    if g['checkin_time'] % 24 < self.checkin_hour:
                        self.metrics['early_checkins'] += 1
        else:
            self.rooms_VC.add(room_id)
        self.maybe_start_hk()

    def run(self):
        # Reset RNG for reproducibility per run
        self._rng.seed(self.random_seed)
        self.init_arrivals()
        while self.event_q:
            t, eid, etype, payload = heapq.heappop(self.event_q)
            if t > self.T_end:
                break
            self.record_time_integrals(t)
            self.time = t
            if etype == 'arrival':
                self.handle_arrival()
            elif etype == 'fd_done':
                self.handle_fd_done(payload)
            elif etype == 'checkout':
                self.handle_checkout(payload)
            elif etype == 'clean_done':
                self.handle_clean_done(payload)
            self.maybe_start_fd()
            self.maybe_start_hk()
        return self.summarize()

    def summarize(self):
        def avg(xs):
            return sum(xs)/len(xs) if xs else 0.0

        fd_wait_avg = avg(self.metrics['fd_wait_times'])
        room_wait_avg = avg(self.metrics['room_wait_times'])
        total_wait_avg = avg(self.metrics['total_to_room_times'])
        early_rate = (self.metrics['early_checkins']/self.metrics['eligible_early']) if self.metrics['eligible_early'] > 0 else 0.0
        avg_fd_q = avg([q for _,q in self.metrics['fd_queue_obs']])
        avg_hk_q = avg([q for _,q in self.metrics['cleaning_queue_obs']])
        avg_occ = avg([o for _,o in self.metrics['occ_obs']])
        occ_rate = avg_occ / self.n_rooms if self.n_rooms>0 else 0.0

        # Utilization (time-weighted capacity approx.)
        if self.metrics['fd_queue_obs']:
            times = [t for t,_ in self.metrics['fd_queue_obs']]
            times.append(self.warmup_days*24)
            times.append(self.total_days*24)
            times = sorted(set(times))
            fd_avail_int = 0.0
            hk_avail_int = 0.0
            for i in range(len(times)-1):
                t0, t1 = times[i], times[i+1]
                tm = 0.5*(t0+t1)
                fd_avail_int += self.fd_agents(tm) * (t1-t0)
                hk_avail_int += self.hk_cleaners(tm) * (t1-t0)
            fd_util = (self.metrics['fd_busy_time']/fd_avail_int) if fd_avail_int>0 else 0.0
            hk_util = (self.metrics['hk_busy_time']/hk_avail_int) if hk_avail_int>0 else 0.0
        else:
            fd_util = hk_util = 0.0

        return {
            'assumptions': {
                'rooms': self.n_rooms,
                'mean_daily_arrivals': self.mean_daily_arrivals,
                'avg_los_nights': self.avg_los_nights,
                'checkin_hour': self.checkin_hour,
                'checkout_hour': self.checkout_hour,
                'front_desk_schedule': 'custom' if self.fd_agents.__name__ != '<lambda>' else 'custom',
                'housekeeping_cleaners_schedule': 'custom' if self.hk_cleaners.__name__ != '<lambda>' else 'custom',
                'sim_days': self.sim_days,
                'warmup_days': self.warmup_days,
                'random_seed': self.random_seed,
            },
            'results (averages over measured window)': {
                'avg_front_desk_wait_minutes': fd_wait_avg * 60.0,
                'avg_wait_for_room_after_fd_minutes': room_wait_avg * 60.0,
                'avg_total_arrival_to_room_minutes': total_wait_avg * 60.0,
                'front_desk_utilization': fd_util,
                'housekeeping_utilization': hk_util,
                'avg_front_desk_queue_len': avg_fd_q,
                'avg_housekeeping_queue_len': avg_hk_q,
                'avg_occupancy_rate': occ_rate,
                'early_checkin_success_rate_given_eligible': early_rate,
                'num_guests_measured': len(self.metrics['total_to_room_times'])
            }
        }

if __name__ == '__main__':
    m = HotelDES2()
    print(m.run())
