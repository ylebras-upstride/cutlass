/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines a densely packed quaternion object intended for storing data in registers and
    executing quaternion operations within a CUDA or host thread.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/vector.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Quaternion: xi + yj + zk + w
template <
  typename Element_ = float      ///< element type
>
class Quaternion : public Array<Element_, 4> {
public:

  /// Logical rank of tensor index space
  static int const kRank = 1;

  /// Number of elements
  static int const kExtent = 4;

  /// Base class is a four-element array
  using Base = Array<Element_, kExtent>;

  /// Element type
  using Element = typename Base::Element;

  /// Reference type to an element
  using Reference = typename Base::reference;

  /// Index type
  using Index = int;

  /// Quaternion storage - imaginary part
  static int const kX = 0;

  /// Quaternion storage - imaginary part
  static int const kY = 1;

  /// Quaternion storage - imaginary part
  static int const kZ = 2;

  /// Quaternion storage - real part
  static int const kW = 3;

public:

  //
  // Methods
  //

  /// Constructs a quaternion
  CUTLASS_HOST_DEVICE
  Quaternion(
    Element w_ = Element(0)
  ) {
    Base::at(kX) = Element(0);
    Base::at(kY) = Element(0);
    Base::at(kZ) = Element(0);
    Base::at(kW) = w_;
  }

  /// Constructs a quaternion
  CUTLASS_HOST_DEVICE
  Quaternion(
    Element x_,
    Element y_,
    Element z_,
    Element w_
  ) {
    Base::at(kX) = x_;
    Base::at(kY) = y_;
    Base::at(kZ) = z_;
    Base::at(kW) = w_;
  }

  /// Constructs a quaternion from a vector representing the imaginary part and a real number
  CUTLASS_HOST_DEVICE
  Quaternion(
    Matrix3x1<Element> const &imag_,
    Element w_ = Element()
  ) {
    Base::at(kX) = imag_[0];
    Base::at(kY) = imag_[1];
    Base::at(kZ) = imag_[2];
    Base::at(kW) = w_;
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference at(Index idx) const {
    return Base::at(idx);
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference at(Index idx) {
    return Base::at(idx);
  }

  /// Accesses the x element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Element x() const {
    return Base::at(kX);
  }

  /// Accesses the x element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Reference x() {
    return Base::at(kX);
  }

  /// Accesses the y element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Element y() const {
    return Base::at(kY);
  }

  /// Accesses the y element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Reference y() {
    return Base::at(kY);
  }

  /// Accesses the z element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Element z() const {
    return Base::at(kZ);
  }

  /// Accesses the z element of the imaginary part of the quaternion
  CUTLASS_HOST_DEVICE
  Reference z() {
    return Base::at(kZ);
  }

  /// Accesses the real part of the quaternion
  CUTLASS_HOST_DEVICE
  Element w() const {
    return Base::at(kW);
  }

  /// Accesses the real part of the quaternion
  CUTLASS_HOST_DEVICE
  Reference w() {
    return Base::at(kW);
  }

  /// Returns the pure imaginary part of the quaternion as a 3-vector
  CUTLASS_HOST_DEVICE
  Matrix3x1<Element> pure() const {
    return Matrix3x1<Element>(x(), y(), z());
  }

  /// Returns a quaternion representation of a spatial rotation given a unit-length axis and
  /// a rotation in radians.
  CUTLASS_HOST_DEVICE
  static Quaternion<Element> rotation(
    Matrix3x1<Element> const &axis_unit,    ///< axis of rotation (assumed to be unit length)
    Element theta) {                        ///< angular rotation in radians

    Element s = fast_sin(theta / Element(2));

    return Quaternion(
      s * axis_unit[0],
      s * axis_unit[1],
      s * axis_unit[2],
      fast_cos(theta / Element(2))
    );
  }
  
  /// Returns a quaternion representation of a spatial rotation represented as a
  /// unit-length rotation axis (r_x, r_y, r_z) and an angular rotation in radians
  CUTLASS_HOST_DEVICE
  static Quaternion<Element> rotation(
    Element r_x,
    Element r_y,
    Element r_z,
    Element theta) {                      ///< angular rotation in radians

    return rotation({r_x, r_y, r_z}, theta);
  }

  /// Geometric rotation of a 3-element vector
  CUTLASS_HOST_DEVICE
  Matrix3x1<Element> rotate(Matrix3x1<Element> const &rhs) const {
    return (*this * Quaternion<Element>(rhs, 0) * reciprocal(*this)).pure();
  }

  /// Inverse rotation operation
  CUTLASS_HOST_DEVICE
  Matrix3x1<Element> rotate_inv(Matrix3x1<Element> const &rhs) const {
    return (reciprocal(*this) * Quaternion<Element>(rhs, 0) * *this).pure();
  }

  /// Rotates a 3-vector assuming this is a unit quaternion (a spinor)
  CUTLASS_HOST_DEVICE
  Matrix3x1<Element> spinor(Matrix3x1<Element> const &rhs) const {
    return (*this * Quaternion<Element>(rhs, 0) * conj(*this)).pure();
  }

  /// Inverse rotation of 3-vector assuming this is a unit quaternion (a spinor)
  CUTLASS_HOST_DEVICE
  Matrix3x1<Element> spinor_inv(Matrix3x1<Element> const &rhs) const {
    return (conj(*this) * Quaternion<Element>(rhs, 0) * *this).pure();
  }

  /// In-place addition
  template <typename Element>
  CUTLASS_HOST_DEVICE 
  Quaternion<Element> &operator+=(Quaternion<Element> const &rhs) {
    *this = (*this + rhs);
    return *this;
  }

  /// In-place subtraction
  template <typename Element>
  CUTLASS_HOST_DEVICE
  Quaternion<Element> &operator-=(Quaternion<Element> const &rhs) {
    *this = (*this - rhs);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  Quaternion<Element> &operator*=(Quaternion<Element> const &rhs) {
    *this = (*this * rhs);
    return *this;
  }

  /// Scalar multiplication
  template <typename T>
  CUTLASS_HOST_DEVICE
  Quaternion<Element> &operator*=(Element s) {
    *this = (*this * s);
    return *this;
  }

  /// In-place Division
  template <typename T>
  CUTLASS_HOST_DEVICE
  Quaternion<Element> &operator/=(Quaternion<Element> const &rhs) {
    *this = (*this / rhs);
    return *this;
  }

  /// In-place Division
  template <typename T>
  CUTLASS_HOST_DEVICE
  Quaternion<Element> &operator/=(Element s) {
    *this = (*this / s);
    return *this;
  }

  /// Computes a 3x3 rotation matrix (row-major representation)
  CUTLASS_HOST_DEVICE
  Matrix3x3<Element> as_rotation_matrix_3x3() const {
    Matrix3x3<Element> m(
      w() * w() + x() * x() - y() * y() - z() * z(),
      2 * x() * y() - 2 * w() * z(),
      2 * x() * z() + 2 * w() * y(),

      2 * x() * y() + 2 * w() * z(),
      w() * w() - x() * x() + y() * y() - z() * z(),
      2 * y() * z() - 2 * w() * x(),

      2 * x() * z() - 2 * w() * y(),
      2 * y() * z() + 2 * w() * x(),
      w() * w() - x() * x() - y() * y() + z() * z()
    );
    return m;
  }

  /// Computes a 4x4 rotation matrix (row-major representation)
  CUTLASS_HOST_DEVICE
  Matrix4x4<Element> as_rotation_matrix_4x4() const {
    Matrix4x4<Element> m = Matrix4x4<Element>::identity();
    m.set_slice_3x3(as_rotation_matrix_3x3());
    return m;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs a quaternion that is non-zero only in its real element.
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> make_Quaternion(
  Element w) {                                ///< real part

  return Quaternion<Element>(w);
}

/// Constructs a quaternion from a vector and real
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> make_Quaternion(
  Matrix3x1<Element> const &imag,             ///< imaginary party as a vector
  Element w) {                                ///< real part

  return Quaternion<Element>(imag, w);
}

/// Constructs a quaternion from a unit-length rotation axis and a rotation 
/// angle in radians
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> make_QuaternionRotation(
  Matrix3x1<Element> const &axis_unit,        ///< rotation axis (unit-length)
  Element w) {                                ///< rotation angle in radians

  return Quaternion<Element>::rotation(axis_unit, w);
}

/// Constructs a quaternion q = xi + yj + zk + w
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> make_Quaternion(Element x, Element y, Element z, Element w) {
  return Quaternion<Element>(x, y, z, w);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns the magnitude of the complex number
template <typename Element>
CUTLASS_HOST_DEVICE
Element abs(Quaternion<Element> const &q) {
  return fast_sqrt(norm(q));
}

/// Quaternion conjugate
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> conj(Quaternion<Element> const &q) {
  return make_Quaternion(
    -q.x(),
    -q.y(),
    -q.z(),
    q.w()
  );
}

/// Computes the squared magnitude of the quaternion
template <typename Element>
CUTLASS_HOST_DEVICE
Element norm(Quaternion<Element> const &q) {
  return q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w();
}

/// Quaternion reciprocal
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> reciprocal(Quaternion<Element> const &q) {
  
  Element nsq = norm(q);
  
  return make_Quaternion(
    -q.x() / nsq,
    -q.y() / nsq,
    -q.z() / nsq,
    q.w() / nsq
  );
}

/// Returns a unit-length quaternion
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> unit(Quaternion<Element> const &q) {
  
  Element rcp_mag = Element(1) / abs(q);
  
  return make_Quaternion(
    q.x() * rcp_mag,
    q.y() * rcp_mag,
    q.z() * rcp_mag,
    q.w() * rcp_mag
  );
}

/// Quaternion exponential
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> exp(Quaternion<Element> const &q) {
  
  Element exp_ = fast_exp(q.w());
  Element imag_norm = fast_sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  Element sin_norm = fast_sin(imag_norm);

  return make_Quaternion(
    exp_ * q.x() * sin_norm / imag_norm,
    exp_ * q.y() * sin_norm / imag_norm,
    exp_ * q.z() * sin_norm / imag_norm,
    exp_ * fast_cos(imag_norm)
  );
}

/// Quaternion natural logarithm
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> log(Quaternion<Element> const &q) {
  
  Element v = fast_sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  Element s = fast_acos(q.w() / abs(q)) / v;
  
  return make_Quaternion(
    q.x() * s,
    q.y() * s,
    q.z() * s,
    fast_log(q.w())
  );
}

/// Gets the rotation angle from a unit-length quaternion
template <typename Element>
CUTLASS_HOST_DEVICE
Element get_rotation_angle(Quaternion<Element> const &q_unit) {
  return fast_acos(q_unit.w()) * Element(2);
}

/// Gets the rotation axis from a unit-length quaternion
template <typename Element>
CUTLASS_HOST_DEVICE
Matrix3x1<Element> get_rotation_axis(Quaternion<Element> const &q_unit) {
  return q_unit.pure().unit();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Equality operator
template <typename Element>
CUTLASS_HOST_DEVICE 
bool operator==(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return lhs.x() == rhs.x() &&
    lhs.y() == rhs.y() &&
    lhs.z() == rhs.z() &&
    lhs.w() == rhs.w();
}

/// Inequality operator
template <typename Element>
CUTLASS_HOST_DEVICE 
bool operator!=(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return !(lhs == rhs);
}

/// Quaternion scalar multiplication
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator*(Quaternion<Element> q, Element s) {
  return make_Quaternion(
    q.x() * s,
    q.y() * s,
    q.z() * s,
    q.w() * s
  );
}

/// Quaternion scalar multiplication
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator*(Element s, Quaternion<Element> const &q) {
  return make_Quaternion(
    s * q.x(),
    s * q.y(),
    s * q.z(),
    s * q.w()
  );
}

/// Quaternion scalar division
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator/(Quaternion<Element> const &q, Element s) {
  return make_Quaternion(
    q.x() / s,
    q.y() / s,
    q.z() / s,
    q.w() / s
  );
}

/// Quaternion unary negation
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator-(Quaternion<Element> const &q) {
  return make_Quaternion(
    -q.x(),
    -q.y(),
    -q.z(),
    -q.w()
  );
}

/// Quaternion addition
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator+(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return make_Quaternion(
    lhs.x() + rhs.x(), 
    lhs.y() + rhs.y(), 
    lhs.z() + rhs.z(), 
    lhs.w() + rhs.w()
  );
}

/// Quaternion subtraction
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator-(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return make_Quaternion(
    lhs.x() - rhs.x(), 
    lhs.y() - rhs.y(), 
    lhs.z() - rhs.z(), 
    lhs.w() - rhs.w()
  );
}

/// Quaternion product
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator*(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return make_Quaternion(
    lhs.w() * rhs.x() + rhs.w() * lhs.x() + lhs.y() * rhs.z() - lhs.z() * rhs.y(),
    lhs.w() * rhs.y() + rhs.w() * lhs.y() + lhs.z() * rhs.x() - lhs.x() * rhs.z(),
    lhs.w() * rhs.z() + rhs.w() * lhs.z() + lhs.x() * rhs.y() - lhs.y() * rhs.x(),
    lhs.w() * rhs.w() - lhs.x() * rhs.x() - lhs.y() * rhs.y() - lhs.z() * rhs.z()
  );
}

/// Quaternion division
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator/(Quaternion<Element> const &lhs, Quaternion<Element> const &rhs) {
  return lhs * reciprocal(rhs);
}

/// Quaternion scalar division
template <typename Element>
CUTLASS_HOST_DEVICE
Quaternion<Element> operator/(Element s, Quaternion<Element> const &q) {
  return s * reciprocal(q);
}

/// Rotates a 3-vector assuming this is a unit quaternion (a spinor). This avoids computing
/// a reciprocal.
template <typename Element>
CUTLASS_HOST_DEVICE
Matrix3x1<Element> spinor_rotation(
  Quaternion<Element> const &spinor,        /// unit-length quaternion
  Matrix3x1<Element> const &rhs) {          /// arbitrary 3-vector

  return (spinor * Quaternion<Element>(rhs, 0) * conj(spinor)).pure();
}

/// Inverse rotation of 3-vector assuming this is a unit quaternion (a spinor). This avoids computing
/// a reciprocal.
template <typename  Element>
CUTLASS_HOST_DEVICE
Matrix3x1<Element> spinor_rotation_inv(
  Quaternion<Element> const &spinor,        /// unit-length quaternion
  Matrix3x1<Element> const &rhs) {          /// arbitrary 3-vector

  return (conj(spinor) * Quaternion<Element>(rhs, 0) * spinor).pure();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Output operators
//

template <typename Element>
std::ostream &operator<<(std::ostream &out, Quaternion<Element> const &q) {
  return out << q.w() << "+i" << q.x() << "+j" << q.y() << "+k" << q.z();
}


template <typename Element>
std::istream &operator>>(std::istream &in, Quaternion<Element> &q) {
  return in >> q.w() >> q.x() >> q.y() >> q.z();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

