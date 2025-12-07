from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base

class VehicleDB(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, unique=True, index=True, nullable=False)
    model = Column(String, nullable=False)
    owner_name = Column(String, nullable=False)
    owner_phone = Column(String, nullable=False)
    service_center = Column(String, nullable=False)

    readings = relationship("TelematicsReadingDB", back_populates="vehicle")
    predictions = relationship("PredictionDB", back_populates="vehicle")
    anomalies = relationship("AnomalyDB", back_populates="vehicle")


class TelematicsReadingDB(Base):
    __tablename__ = "telematics_readings"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, ForeignKey("vehicles.vehicle_id"), index=True, nullable=False)
    timestamp = Column(String, nullable=False)
    engine_temp = Column(Float, nullable=False)
    rpm = Column(Float, nullable=False)
    brake_pressure = Column(Float, nullable=False)
    bearing_temp = Column(Float, nullable=False)
    fuel_level = Column(Float, nullable=False)
    odometer = Column(Float, nullable=False)

    vehicle = relationship("VehicleDB", back_populates="readings")


class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, ForeignKey("vehicles.vehicle_id"), index=True, nullable=False)
    risk_score = Column(Float, nullable=False)
    predicted_failure_type = Column(String, nullable=False)
    time_to_failure_days = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicle = relationship("VehicleDB", back_populates="predictions")


class AnomalyDB(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, ForeignKey("vehicles.vehicle_id"), index=True, nullable=False)
    anomaly_type = Column(String, nullable=False)
    sensor = Column(String, nullable=False)
    current_value = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicle = relationship("VehicleDB", back_populates="anomalies")
