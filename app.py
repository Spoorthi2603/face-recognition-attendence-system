import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import traceback
import sys

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing Face Detector
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

#### Global Model Cache
MODEL = None

#### Create directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### Get total registered users
def totalreg():
    try:
        return len(os.listdir('static/faces'))
    except Exception as e:
        print(f"Error in totalreg: {e}")
        return 0


#### Extract faces from image
def extract_faces(img):
    """Extract faces with validation"""
    try:
        if img is None or img.size == 0:
            return np.array([])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(300, 300)
        )
        
        return face_points
    
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return np.array([])


#### Load model (cached globally)
def load_model():
    """Load model once and cache globally"""
    global MODEL
    try:
        if MODEL is None:
            MODEL = joblib.load('static/face_recognition_model.pkl')
            print("✓ Model loaded successfully")
        return MODEL
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


#### Identify face
def identify_face(facearray):
    """Identify person from face array"""
    try:
        if facearray is None or facearray.size == 0:
            return 'Unknown'
        
        if len(facearray.shape) != 2 or facearray.shape[0] != 1:
            return 'Invalid_Input'
        
        model = load_model()
        
        if model is None:
            return 'No_Model'
        
        try:
            expected_features = model._fit_X.shape[1]
        except:
            expected_features = 7500
        
        if facearray.shape[1] != expected_features:
            return 'Shape_Error'
        
        prediction = model.predict(facearray)
        return prediction[0]
    
    except Exception as e:
        print(f"Error in identify_face: {e}")
        return 'Error'


#### Train model
def train_model():
    """Train model with error handling"""
    try:
        faces = []
        labels = []
        
        if not os.path.exists('static/faces'):
            print("Faces directory not found!")
            return False
        
        userlist = os.listdir('static/faces')
        
        if len(userlist) == 0:
            print("No users registered yet!")
            return False
        
        print(f"Training on {len(userlist)} users...")
        
        for user in userlist:
            user_path = f'static/faces/{user}'
            
            if not os.path.isdir(user_path):
                continue
            
            images = os.listdir(user_path)
            
            if len(images) == 0:
                print(f"Warning: User {user} has no images")
                continue
            
            for imgname in images:
                try:
                    img_path = f'{user_path}/{imgname}'
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Could not read: {img_path}")
                        continue
                    
                    if len(img.shape) != 3 or img.shape[0] == 0 or img.shape[1] == 0:
                        print(f"Invalid image shape: {img_path}")
                        continue
                    
                    resized_face = cv2.resize(img, (50, 50))
                    faces.append(resized_face.ravel())
                    labels.append(user)
                
                except Exception as e:
                    print(f"Error processing {imgname}: {str(e)}")
                    continue
        
        if len(faces) == 0:
            print("No valid face data found for training!")
            return False
        
        faces = np.array(faces)
        print(f"Training with {len(faces)} face samples...")
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        
        global MODEL
        MODEL = None
        
        print(f"✓ Model trained successfully!")
        return True
    
    except Exception as e:
        print(f"Error in train_model: {e}")
        traceback.print_exc()
        return False


#### Extract attendance
def extract_attendance():
    """Extract attendance with error handling"""
    try:
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        
        if not os.path.exists(csv_path):
            return [], [], [], 0
        
        df = pd.read_csv(csv_path)
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        l = len(df)
        return names, rolls, times, l
    
    except Exception as e:
        print(f"Error in extract_attendance: {e}")
        return [], [], [], 0


#### Add attendance
def add_attendance(name):
    """Add attendance with proper CSV handling"""
    try:
        if not isinstance(name, str) or '_' not in name:
            return False
        
        parts = name.split('_')
        if len(parts) < 2:
            return False
        
        username = parts[0]
        
        try:
            userid = int(parts[1])
        except ValueError:
            return False
        
        current_time = datetime.now().strftime("%H:%M:%S")
        csv_path = f'Attendance/Attendance-{datetoday}.csv'
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Time\n')
        
        df = pd.read_csv(csv_path)
        
        # Prevent duplicates
        if userid in df['Roll'].values:
            print(f"✓ {username} already marked present")
            return False
        
        # Use pandas append (proper way)
        new_record = pd.DataFrame({
            'Name': [username],
            'Roll': [userid],
            'Time': [current_time]
        })
        
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Attendance added: {username} at {current_time}")
        return True
    
    except Exception as e:
        print(f"Error in add_attendance: {e}")
        return False


################## ROUTING FUNCTIONS #########################

#### Home page
@app.route('/')
def home():
    try:
        names, rolls, times, l = extract_attendance()    
        return render_template('home.html',
            names=names, rolls=rolls, times=times, l=l,
            totalreg=totalreg(), datetoday2=datetoday2)
    except Exception as e:
        print(f"Error in home: {e}")
        return render_template('home.html',
            names=[], rolls=[], times=[], l=0,
            totalreg=totalreg(), datetoday2=datetoday2,
            mess=f'⚠️ Error: {str(e)[:50]}')


#### Take Attendance (✅ FIXED with debugging)
@app.route('/start', methods=['GET'])
def start():
    """
    Take attendance with proper debugging and error handling
    """
    print("\n" + "="*70)
    print("📸 START ROUTE CALLED - BEGIN ATTENDANCE SESSION")
    print("="*70)
    
    cap = None
    
    try:
        # ============ STEP 1: CHECK MODEL ============
        print("\n[STEP 1] Checking if model exists...")
        sys.stdout.flush()
        
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            print("❌ Model file not found!")
            sys.stdout.flush()
            
            msg = '⚠️ No trained model found. Please register users first.'
            print(f"[STEP 5] Returning to home with message: {msg}")
            sys.stdout.flush()
            
            return render_template('home.html',
                names=[], rolls=[], times=[], l=0,
                totalreg=totalreg(), datetoday2=datetoday2, mess=msg)
        
        print("✅ Model file found!")
        sys.stdout.flush()
        
        # ============ STEP 2: OPEN WEBCAM ============
        print("[STEP 2] Opening webcam (VideoCapture)...")
        sys.stdout.flush()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open webcam!")
            sys.stdout.flush()
            
            msg = '⚠️ Webcam not available!'
            print(f"[STEP 5] Returning to home with message: {msg}")
            sys.stdout.flush()
            
            return render_template('home.html',
                names=[], rolls=[], times=[], l=0,
                totalreg=totalreg(), datetoday2=datetoday2, mess=msg)
        
        print("✅ Webcam opened successfully!")
        sys.stdout.flush()
        
        # ============ STEP 3: PROCESS FRAMES ============
        print("[STEP 3] Starting frame processing...")
        sys.stdout.flush()
        
        ret = True
        frame_count = 0
        max_frames = 300  # ~10 seconds
        detected_users = set()
        
        print(f"  Will process up to {max_frames} frames...")
        sys.stdout.flush()
        
        while ret and frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                print(f"  ⚠️ Frame read failed at frame {frame_count}")
                sys.stdout.flush()
                break
            
            frame_count += 1
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                print(f"  Processing frame {frame_count}/{max_frames}...")
                sys.stdout.flush()
            
            # Get faces
            faces = extract_faces(frame)
            
            # Process faces
            if len(faces) > 0:
                try:
                    (x, y, w, h) = faces[0]
                    
                    # Validate coordinates
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        continue
                    
                    # Get ROI
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    
                    # Process face
                    face = cv2.resize(roi, (50, 50))
                    face_flat = face.ravel()
                    
                    # Validate shape
                    if face_flat.shape[0] != 7500:
                        continue
                    
                    # Identify person
                    identified_person = identify_face(face_flat.reshape(1, -1))
                    
                    # Add attendance if valid
                    if identified_person and not identified_person.startswith('Error'):
                        if add_attendance(identified_person):
                            detected_users.add(identified_person)
                            print(f"  ✓ Frame {frame_count}: {identified_person} detected")
                            sys.stdout.flush()
                
                except Exception as e:
                    print(f"  ⚠️ Error processing face: {e}")
                    sys.stdout.flush()
        
        print(f"\n✅ Frame processing completed!")
        print(f"   Total frames processed: {frame_count}/{max_frames}")
        print(f"   Users detected: {len(detected_users)}")
        sys.stdout.flush()
        
        # ============ STEP 4: CLEANUP ============
        print("\n[STEP 4] Cleaning up resources...")
        sys.stdout.flush()
        
        if cap is not None:
            cap.release()
            print("✅ Webcam released")
            sys.stdout.flush()
        
        try:
            cv2.destroyAllWindows()
            print("✅ All OpenCV windows closed")
            sys.stdout.flush()
        except:
            pass
        
        # ============ STEP 5: PREPARE RESPONSE ============
        print("\n[STEP 5] Preparing response...")
        sys.stdout.flush()
        
        print("  Reading attendance file...")
        sys.stdout.flush()
        names, rolls, times, l = extract_attendance()
        
        print(f"  ✓ Attendance file read: {l} records found")
        sys.stdout.flush()
        
        print("  Rendering home.html template...")
        sys.stdout.flush()
        
        response = render_template('home.html',
            names=names, rolls=rolls, times=times, l=l,
            totalreg=totalreg(), datetoday2=datetoday2)
        
        print("✅ home.html rendered successfully!")
        sys.stdout.flush()
        
        print("\n" + "="*70)
        print("✅ SUCCESS - ATTENDANCE SESSION COMPLETED")
        print("="*70 + "\n")
        sys.stdout.flush()
        
        return response
    
    except Exception as e:
        print(f"\n❌ EXCEPTION OCCURRED!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print("   Full traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        # Make sure to release webcam on error
        if cap is not None:
            try:
                cap.release()
                cv2.destroyAllWindows()
                print("✓ Cleanup completed on error")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")
            sys.stdout.flush()
        
        print("\n" + "="*70)
        print("🔴 ERROR - RETURNING ERROR PAGE")
        print("="*70 + "\n")
        sys.stdout.flush()
        
        error_msg = f'⚠️ Error occurred: {str(e)[:100]}'
        
        try:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html',
                names=names, rolls=rolls, times=times, l=l,
                totalreg=totalreg(), datetoday2=datetoday2, mess=error_msg)
        except:
            return render_template('home.html',
                names=[], rolls=[], times=[], l=0,
                totalreg=totalreg(), datetoday2=datetoday2, mess=error_msg)


#### Add new user
@app.route('/add', methods=['GET','POST'])
def add():
    try:
        newusername = request.form.get('newusername', '').strip()
        newuserid = request.form.get('newuserid', '').strip()
        
        # Validate input
        if not newusername or not newuserid:
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess='⚠️ Username and User ID are required!')
        
        # Validate user ID is numeric
        try:
            newuserid = int(newuserid)
        except ValueError:
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess='⚠️ User ID must be a number!')
        
        # Sanitize username
        if '/' in newusername or '\\' in newusername or '..' in newusername:
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess='⚠️ Invalid characters in username!')
        
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        
        # Check if user exists
        if os.path.isdir(userimagefolder):
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess=f'⚠️ User {newusername} already exists!')
        
        os.makedirs(userimagefolder)
        
        # Check webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess='⚠️ Webcam not available!')
        
        i, j = 0, 0
        max_frames = 500
        required_images = 50
        
        print(f"\n📸 Capturing images for user {newusername}_{newuserid}...")
        
        while j < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            faces = extract_faces(frame)
            
            # Capture faces without displaying window
            for (x, y, w, h) in faces:
                if j % 10 == 0 and i < required_images:
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            name = f'{newusername}_{i}.jpg'
                            cv2.imwrite(f'{userimagefolder}/{name}', face_roi)
                            i += 1
                            print(f"  ✓ Captured image {i}/{required_images}")
                    except Exception as e:
                        print(f"Error saving image: {e}")
                
                j += 1
            
            # Check if we have enough images
            if i >= required_images:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Check if we captured enough
        if i < required_images:
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess=f'⚠️ Only captured {i} images. Need {required_images}!')
        
        # Train model
        print(f'Training model with {i} images...')
        if train_model():
            print(f"✓ User {newusername} registered successfully!\n")
            names, rolls, times, l = extract_attendance()
            return render_template('home.html',
                names=names, rolls=rolls, times=times, l=l,
                totalreg=totalreg(), datetoday2=datetoday2,
                mess=f'✓ User {newusername} registered successfully!')
        else:
            return render_template('home.html',
                totalreg=totalreg(),
                datetoday2=datetoday2,
                mess='⚠️ Error training model!')
    
    except Exception as e:
        print(f"Error in add: {e}")
        traceback.print_exc()
        return render_template('home.html',
            totalreg=totalreg(),
            datetoday2=datetoday2,
            mess=f'⚠️ Error: {str(e)[:50]}')


#### Error handler
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Unhandled error: {error}")
    traceback.print_exc()
    return render_template('home.html',
        totalreg=totalreg(),
        datetoday2=datetoday2,
        mess=f'⚠️ An error occurred: {str(error)[:50]}')


#### Run Flask App
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎓 FACE RECOGNITION ATTENDANCE SYSTEM - STARTING")
    print("="*70)
    print(f"Date: {datetoday2}")
    print(f"Total registered users: {totalreg()}")
    print("="*70)
    print("✓ Opening: http://127.0.0.1:5000/")
    print("="*70 + "\n")
    
    app.run(debug=False, use_reloader=False)